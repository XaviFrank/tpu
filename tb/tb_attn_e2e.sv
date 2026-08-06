`timescale 1ns / 1ps
//====================================================================
// tb_attn_e2e - the whole KV path, write through read, on a real row.
//
//   real INT8 K -> kv_pack_int8 ------------> kv_cache_store (32 w)
//   real INT8 V -> kv_rotate -> kv_quantize -> kv_cache_store (13 w)
//                                                   |
//                           read back per position  v
//                                           attn_scan_gqa -> o[], l
//
//   Nothing is injected mid-path. The earlier attention testbenches
//   fed cache words from a file; here the words come out of the store
//   the write path actually wrote, so a disagreement between the two
//   sides about word order, line length or addressing has somewhere to
//   show up. That is the point of an integration test: the unit tests
//   already proved each stage against its own golden, and a SHARED
//   misunderstanding is exactly what per-stage goldens cannot catch.
//
//   Checked against the same independent per-head numpy goldens the
//   GQA testbench uses, so a pass means the assembled hardware
//   reproduces the reference attention rather than merely being
//   self-consistent.
//====================================================================
module tb_attn_e2e;

    localparam int N   = 128;
    localparam int NQ  = 2;
    localparam int T   = 102;
    localparam int NWK = 32;
    localparam int NWV = 13;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    integer errors = 0, i, t, hh, bad;

    logic [7:0]  xk [T*N], xv [T*N];
    logic [31:0] yq0 [N], yq1 [N];
    logic [47:0] eo0 [N], eo1 [N], el0 [1], el1 [1];

    initial begin
        $readmemh("tb/attn_xk.memh",  xk);
        $readmemh("tb/attn_xv.memh",  xv);
        $readmemh("tb/attn_yq0.memh", yq0);
        $readmemh("tb/attn_yq1.memh", yq1);
        $readmemh("tb/attn_o0.memh",  eo0);
        $readmemh("tb/attn_o1.memh",  eo1);
        $readmemh("tb/attn_l0.memh",  el0);
        $readmemh("tb/attn_l1.memh",  el1);
    end

    // ---------------- write path: K ----------------
    logic               kin_v, kp_busy, kp_wv, kp_done;
    logic signed [23:0] kin_d;
    logic [4:0]         kp_wi;
    logic [31:0]        kp_wd;

    kv_pack_int8 #(.N(N)) u_kpack (
        .clk(clk), .rst_n(rst_n), .in_valid(kin_v), .in_data(kin_d),
        .busy(kp_busy), .w_valid(kp_wv), .w_idx(kp_wi), .w_data(kp_wd),
        .done(kp_done));

    // ---------------- write path: V ----------------
    logic [7:0]         vbuf [N];
    logic               vstart, vr_busy, vr_en, vr_done;
    logic [6:0]         vr_ad;
    logic signed [7:0]  vr_dt;
    logic               vy_v;
    logic signed [23:0] vy_d;
    logic               vq_busy, vq_wv, vq_done;
    logic [4:0]         vq_wi;
    logic [31:0]        vq_wd;

    always_ff @(posedge clk) vr_dt <= vbuf[vr_ad];

    kv_rotate #(.N(N), .ROT_MODE(0)) u_vrot (
        .clk(clk), .rst_n(rst_n), .start(vstart), .busy(vr_busy),
        .x_rd_en(vr_en), .x_rd_addr(vr_ad), .x_rd_data(vr_dt),
        .y_valid(vy_v), .y_idx(), .y_out(vy_d), .done(vr_done));

    kv_quantize #(.N(N), .QBITS(3), .QJL_EN(0),
                  .CB_PATH("rtl/kv_cb_m0_b3.memh")) u_vq (
        .clk(clk), .rst_n(rst_n), .in_valid(vy_v), .in_data(vy_d),
        .busy(vq_busy), .w_valid(vq_wv), .w_idx(vq_wi), .w_data(vq_wd),
        .done(vq_done));

    // ---------------- the caches ----------------
    logic [6:0]  kn_pos, vn_pos;
    logic        k_full, k_ovf, v_full, v_ovf;
    logic        krd_en, vrd_en;
    logic [6:0]  krd_pos, vrd_pos;
    logic [4:0]  krd_wi, vrd_wi;
    logic [31:0] krd_d, vrd_d;

    kv_cache_store #(.WORDS_PER_VEC(NWK), .MAX_POS(T)) u_kstore (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .w_valid(kp_wv), .w_idx(kp_wi), .w_data(kp_wd), .commit(kp_done),
        .n_pos(kn_pos), .full(k_full), .overflow(k_ovf),
        .rd_en(krd_en), .rd_pos(krd_pos), .rd_widx(krd_wi), .rd_data(krd_d));

    kv_cache_store #(.WORDS_PER_VEC(NWV), .MAX_POS(T)) u_vstore (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .w_valid(vq_wv), .w_idx(vq_wi), .w_data(vq_wd), .commit(vq_done),
        .n_pos(vn_pos), .full(v_full), .overflow(v_ovf),
        .rd_en(vrd_en), .rd_pos(vrd_pos), .rd_widx(vrd_wi), .rd_data(vrd_d));

    // ---------------- read path ----------------
    logic q_wr, prep_start, prep_done, pos_last, s_busy, pos_done;
    logic q_head, o_rd_head;
    logic [6:0] q_addr, o_rd_addr;
    logic signed [31:0] q_data;
    logic kfv, vfv;
    logic signed [47:0] o_rd_data;
    logic [47:0] l_out;
    logic signed [15:0] m_out;

    // the store's read port is registered, so the scan's word-valid is
    // simply the read enable delayed one cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kfv <= 1'b0;
            vfv <= 1'b0;
        end else begin
            kfv <= krd_en;
            vfv <= vrd_en;
        end
    end

    attn_scan_gqa #(.N(N), .NQ(NQ), .SSHIFT(0)) u_scan (
        .clk(clk), .rst_n(rst_n),
        .q_wr(q_wr), .q_head(q_head), .q_addr(q_addr), .q_data(q_data),
        .prep_start(prep_start), .prep_done(prep_done),
        .pos_last(pos_last), .busy(s_busy), .pos_done(pos_done),
        .k_w_valid(kfv), .k_w_data(krd_d),
        .v_w_valid(vfv), .v_w_data(vrd_d),
        .o_rd_head(o_rd_head), .o_rd_addr(o_rd_addr),
        .o_rd_data(o_rd_data), .l_out(l_out), .m_out(m_out));

    initial begin
        kin_v = 0; kin_d = 0; vstart = 0; q_wr = 0; prep_start = 0;
        pos_last = 0; krd_en = 0; vrd_en = 0; krd_pos = 0; vrd_pos = 0;
        krd_wi = 0; vrd_wi = 0; q_head = 0; o_rd_head = 0; q_addr = 0;
        o_rd_addr = 0; q_data = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        $display("=== attn_e2e: %0d positions through the real write path ===", T);

        // ---------- WRITE ----------
        for (t = 0; t < T; t = t + 1) begin
            for (i = 0; i < N; i = i + 1) vbuf[i] = xv[t*N + i];
            @(negedge clk);
            vstart = 1'b1;
            @(negedge clk);
            vstart = 1'b0;
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                kin_v = 1'b1;
                kin_d = {{16{xk[t*N+i][7]}}, xk[t*N+i]};
            end
            @(negedge clk);
            kin_v = 1'b0;
            while (kp_busy || vr_busy || vq_busy) @(negedge clk);
            repeat (2) @(negedge clk);
        end
        if (kn_pos !== T[6:0] || vn_pos !== T[6:0] || k_ovf || v_ovf) begin
            errors = errors + 1;
            $display("Error: E1 stores hold k %0d v %0d, overflow %b%b, expected %0d",
                     kn_pos, vn_pos, k_ovf, v_ovf, T);
        end
        $display("  E1 wrote %0d K lines and %0d V lines, no overflow",
                 kn_pos, vn_pos);

        // ---------- READ ----------
        for (hh = 0; hh < NQ; hh = hh + 1)
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                q_wr   = 1'b1;
                q_head = hh[0];
                q_addr = i[6:0];
                q_data = (hh == 0) ? yq0[i] : yq1[i];
            end
        @(negedge clk);
        q_wr = 1'b0;
        @(negedge clk); prep_start = 1'b1;
        @(negedge clk); prep_start = 1'b0;
        @(posedge prep_done);
        repeat (2) @(negedge clk);

        for (t = 0; t < T; t = t + 1) begin
            pos_last = (t == T-1);
            while (s_busy) @(negedge clk);
            krd_pos = t[6:0];
            for (i = 0; i < NWK; i = i + 1) begin
                @(negedge clk);
                krd_en = 1'b1;
                krd_wi = i[4:0];
            end
            @(negedge clk);
            krd_en = 1'b0;
            vrd_pos = t[6:0];
            for (i = 0; i < NWV; i = i + 1) begin
                @(negedge clk);
                vrd_en = 1'b1;
                vrd_wi = i[4:0];
            end
            @(negedge clk);
            vrd_en = 1'b0;
            @(posedge pos_done);
            repeat (2) @(negedge clk);
        end
        pos_last = 1'b0;
        repeat (4) @(negedge clk);

        // ---------- CHECK ----------
        for (hh = 0; hh < NQ; hh = hh + 1) begin
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                o_rd_head = hh[0];
                o_rd_addr = i[6:0];
                @(negedge clk);
                if (o_rd_data !== $signed((hh == 0) ? eo0[i] : eo1[i])) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: E2 head %0d o[%0d] got %012h exp %012h",
                                 hh, i, o_rd_data, (hh == 0) ? eo0[i] : eo1[i]);
                end
            end
            errors = errors + bad;
            @(negedge clk); o_rd_head = hh[0]; @(negedge clk);
            if (l_out !== ((hh == 0) ? el0[0] : el1[0])) begin
                errors = errors + 1;
                $display("Error: E2 head %0d l got %012h exp %012h",
                         hh, l_out, (hh == 0) ? el0[0] : el1[0]);
            end
            $display("  E2 head %0d: %0d/%0d coords bit-exact vs numpy, l %0d",
                     hh, N - bad, N, l_out);
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #200000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
