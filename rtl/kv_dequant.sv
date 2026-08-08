`timescale 1ns / 1ps
//====================================================================
// kv_dequant: unpack, reconstruct, inverse-rotate. Step 5 of the KV
// cache write path.
//
//   Built for VERIFICATION, not for the attention read side. Its job
//   is to make the write path falsifiable: without a way to read back
//   what was written, "the cache stores K and V correctly" is not a
//   claim anything can test. The real read side is streaming attention
//   over the cache, which is a separate spec and does not reconstruct
//   x at all - it consumes the quantized form directly.
//
//====================================================================
// INVERSE SCALE - the forward path shed it, so this puts it back
//   kv_rotate deliberately carries no scale, which is what let the
//   forward goldens be integer-exact. The inverse has to restore it:
//
//     ROT_MODE 0   H is symmetric with H*H = N*I, so the inverse is
//                  the SAME butterfly, then D, then divide by N.
//                  N is a power of two, so INV_MUL is a power of two
//                  and the multiply degenerates to a shift. This path
//                  is EXACTLY invertible - the only loss in the whole
//                  round trip is the quantizer.
//     ROT_MODE 1   Pi_int8 is only APPROXIMATELY orthogonal - that is
//                  the residual measured back in step 1 - so the
//                  inverse is Pi^T followed by a divide by
//                  c = mean row norm squared. c is not a power of two,
//                  hence a genuine constant multiplier, and the
//                  approximation is a second error term the Hadamard
//                  path simply does not have.
//
//   Note the sign flip moves. Forward is y = H*(D*x), so the inverse
//   is x = D*(H*y)/N: butterfly FIRST, D after. Applying D on the
//   wrong side is a bug that leaves norms and Parseval intact and only
//   shows up against real values, which is why the round trip compares
//   values rather than statistics.
//
//====================================================================
// QJL RECONSTRUCTION
//   The residual is stored as one sign bit per coordinate plus its
//   norm. Reconstruction spreads that norm evenly across coordinates,
//   amp = ||r||/sqrt(N), and applies the stored sign. That is the
//   natural 1-bit-per-coordinate estimate: it restores the residual's
//   ENERGY and its DIRECTION per coordinate, which is what removes the
//   MSE quantizer's inner-product bias.
//
//   Whether it removes that bias in the sense the paper proves is a
//   claim about inner products, and inner products live on the read
//   side. This module can only show the reconstruction is faithful.
//====================================================================
module kv_dequant #(
    parameter int N         = 128,
    parameter int QBITS     = 3,
    parameter int QJL_EN    = 0,
    parameter int ROT_MODE  = 0,
    parameter int INV_MUL   = 8388608,
    parameter int INV_SHIFT = 30,
    parameter CB_PATH       = "rtl/kv_cb_m0_b3.memh",
    parameter ROT_SIGN_PATH = "rtl/kv_rot_sign.memh",
    parameter ROT_PI_PATH   = "rtl/kv_rot_pi.memh"
)(
    input  logic clk,
    input  logic rst_n,

    // packed cache line in, same word order kv_quantize emitted
    input  logic        w_valid,
    input  logic [31:0] w_data,
    output logic        busy,

    // reconstructed INT8 vector out
    output logic                 o_valid,
    output logic [$clog2(N)-1:0] o_idx,
    output logic signed [7:0]    o_data,
    output logic                 done
);

    localparam int AW     = $clog2(N);
    localparam int NLEV   = 1 << QBITS;
    localparam int NWQ    = (N * QBITS) / 32;
    localparam int NWORDS = (QJL_EN != 0) ? (NWQ + 6) : (NWQ + 1);
    localparam int QB     = N * QBITS;
    localparam int ACC_W  = 48;
    localparam int ROOT_W = 30;

    localparam logic [AW:0]   N_LAST  = N - 1;
    // round(32768/sqrt(N)); N=128 gives 2896.
    localparam int INV_SQRTN = 2896;

    (* rom_style = "distributed" *) logic [15:0] cb_rom [2*NLEV-1];
    initial $readmemh(CB_PATH, cb_rom);

    typedef enum logic [2:0] {
        D_IDLE, D_LEV, D_REC, D_ROT, D_OUT
    } state_t;
    state_t state;

    logic [AW:0]        cnt;
    logic [QB-1:0]      qbits;
    logic [N-1:0]       sbits;
    logic [ROOT_W-1:0]  vnorm, rnorm;
    logic               entry_valid, u_busy;
    logic [31:0]        lev_s [NLEV];
    logic [31:0]        amp;
    logic               calc_done;
    logic [ACC_W-1:0]   out_data;

    assign busy = (state != D_IDLE) || u_busy;

    //----------------------------------------------------------------
    // Word intake is kv_unpack's job, shared with the attention scan.
    // This module adds the level rescale and the inverse rotation,
    // which are the parts the scan deliberately does NOT want - see
    // kv_unpack.sv's header for why that boundary is where it is.
    //----------------------------------------------------------------
    kv_unpack #(.N(N), .QBITS(QBITS), .QJL_EN(QJL_EN)) u_unpack (
        .clk         (clk),
        .rst_n       (rst_n),
        .w_valid     (w_valid),
        .w_data      (w_data),
        .busy        (u_busy),
        .codes       (qbits),
        .signs       (sbits),
        .vnorm       (vnorm),
        .rnorm       (rnorm),
        .entry_valid (entry_valid)
    );

    //----------------------------------------------------------------
    // Rescale the codebook levels by ||y||, exactly as kv_quantize did.
    //----------------------------------------------------------------
    logic [4:0]         lvi;
    logic signed [15:0] cb_cur;
    logic signed [47:0] cb_prod;
    logic [31:0]        cb_scaled;
    logic signed [47:0] amp_prod;

    assign lvi       = cnt[4:0] + NLEV[4:0] - 5'd1;
    assign cb_cur    = cb_rom[lvi];
    assign cb_prod   = cb_cur * $signed({1'b0, vnorm});
    assign cb_scaled = cb_prod[46:15];
    assign amp_prod  = $signed({1'b0, rnorm}) * INV_SQRTN;

    always_ff @(posedge clk) begin
        if (state == D_LEV) begin
            lev_s[cnt[4:0]] <= cb_scaled;
            amp             <= amp_prod[46:15];
        end
    end

    //----------------------------------------------------------------
    // Reconstruct y_hat, one coordinate per cycle.
    //----------------------------------------------------------------
    logic [AW-1:0]      ridx;
    logic [QBITS-1:0]   qi;
    logic               si;
    logic [31:0]        lev_cur;
    logic signed [31:0] yhat;

    assign ridx    = cnt[AW-1:0];
    assign qi      = qbits[(N - 1 - ridx) * QBITS +: QBITS];
    assign si      = sbits[N - 1 - ridx];
    assign lev_cur = lev_s[qi];

    always_comb begin
        if (QJL_EN == 0)  yhat = $signed(lev_cur);
        else if (si)      yhat = $signed(lev_cur) - $signed(amp);
        else              yhat = $signed(lev_cur) + $signed(amp);
    end

    //----------------------------------------------------------------
    // Final scale and saturate.
    //----------------------------------------------------------------
    logic signed [79:0] scaled;
    logic signed [79:0] shifted;
    logic        [7:0]  shift_lo;

    assign scaled   = $signed(out_data) * INV_MUL;
    assign shifted  = scaled >>> INV_SHIFT;
    assign shift_lo = shifted[7:0];   // hoisted: Icarus half-supports
                                      // a part-select inside always_comb

    // REGISTERED, like o_valid and o_idx. Leaving the saturate
    // combinational while the valid and index are registered puts the
    // data one cycle ahead of its own index - the value for
    // coordinate k arrives labelled k-1. kv_rotate registers all three
    // together for the same reason.
    logic signed [7:0] o_data_c;

    always_comb begin
        if      (shifted >  80'sd127) o_data_c = 8'sd127;
        else if (shifted < -80'sd128) o_data_c = -8'sd128;
        else                          o_data_c = shift_lo;
    end

    always_ff @(posedge clk) o_data <= o_data_c;

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= D_IDLE;
            cnt   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                D_IDLE: if (entry_valid) begin
                    state <= D_LEV;
                    cnt   <= '0;
                end
                D_LEV: if (cnt == NLEV - 1) begin
                    state <= D_REC;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                D_REC: if (cnt == N_LAST) begin
                    state <= D_ROT;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                D_ROT: if (calc_done) begin
                    state <= D_OUT;
                    cnt   <= '0;
                end
                D_OUT: if (cnt == N_LAST) begin
                    state <= D_IDLE;
                    cnt   <= '0;
                    done  <= 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                default: state <= D_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) o_valid <= 1'b0;
        else        o_valid <= (state == D_OUT);
    end

    always_ff @(posedge clk) o_idx <= cnt[AW-1:0];

    //----------------------------------------------------------------
    // The inverse rotation is kv_irotate's job, shared with the
    // attention output path. `run` is literally the old
    // `state == D_ROT`, and the read port is combinational with D
    // already applied, so the scale-and-saturate below is unchanged.
    //----------------------------------------------------------------
    logic               rot_run;
    logic [AW-1:0]      rot_rd_idx;
    logic               rec_wr;
    logic [ACC_W-1:0]   yhat_ext;

    assign rot_run    = (state == D_ROT);
    assign rot_rd_idx = cnt[AW-1:0];
    assign rec_wr     = (state == D_REC);
    assign yhat_ext   = {{(ACC_W-32){yhat[31]}}, yhat};

    kv_irotate #(.N(N), .DW(ACC_W), .ROT_MODE(ROT_MODE),
                 .ROT_SIGN_PATH(ROT_SIGN_PATH),
                 .ROT_PI_PATH(ROT_PI_PATH)) u_irot (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (rec_wr),
        .in_idx    (ridx),
        .in_data   (yhat_ext),
        .run       (rot_run),
        .calc_done (calc_done),
        .rd_idx    (rot_rd_idx),
        .rd_data   (out_data)
    );

endmodule
