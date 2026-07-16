`timescale 1ns / 1ps
//====================================================================
// axi_lite_ctrl v1 (AXI spec step 2): AXI-Lite control/status
// skeleton, CTRL/STATUS only, SINGLE SHARED CLOCK assumption -
// core_clk must be tied to s_axi_aclk until the CDC step lands (the
// port split exists so that step only touches internals).
//
// Register map (spec section 3 contract; rest lands in step 4):
//   0x00 CTRL    W  [0] go - self-clearing pulse, also clears done
//   0x04 STATUS  R  [0] busy, [1] done (sticky; set when busy falls)
//
// Writes: AW and W accepted independently (either order), commit
// when both are held, single outstanding response. wstrb ignored
// (control plane, full-word writes assumed). Reads: single
// outstanding, registered rdata, held until rready. Resps all OKAY.
// Unmapped/write-only reads return 0.
//====================================================================
module axi_lite_ctrl (
    // AXI-Lite subordinate (PS domain)
    input  logic        s_axi_aclk,
    input  logic        s_axi_aresetn,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [5:0]  s_axi_awaddr,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    output logic [1:0]  s_axi_bresp,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    input  logic [5:0]  s_axi_araddr,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,

    // core-domain side (v1: tie core_clk to s_axi_aclk externally)
    input  logic        core_clk,
    input  logic        core_rst_n,
    output logic        core_go,
    input  logic        core_busy
);

    // word address = byte address [5:2]
    localparam logic [3:0] A_CTRL   = 4'h0;
    localparam logic [3:0] A_STATUS = 4'h1;

    //----------------------------------------------------------------
    // Write channel: capture AW / W independently, commit together
    //----------------------------------------------------------------
    logic        aw_got, w_got;
    logic [3:0]  waddr_q;
    logic [31:0] wdata_q;
    logic        commit;
    logic        go_write;

    assign s_axi_awready = !aw_got && !s_axi_bvalid;
    assign s_axi_wready  = !w_got  && !s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;
    assign commit        = aw_got && w_got && !s_axi_bvalid;
    assign go_write      = commit && (waddr_q == A_CTRL) && wdata_q[0];

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            aw_got       <= 1'b0;
            w_got        <= 1'b0;
            s_axi_bvalid <= 1'b0;
            core_go      <= 1'b0;
        end else begin
            core_go <= go_write;      // v1 only: same-clock 1-cycle pulse

            if (s_axi_awvalid && s_axi_awready) begin
                waddr_q <= s_axi_awaddr[5:2];
                aw_got  <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_q <= s_axi_wdata;
                w_got   <= 1'b1;
            end
            if (commit) begin
                s_axi_bvalid <= 1'b1;
                aw_got       <= 1'b0;
                w_got        <= 1'b0;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    //----------------------------------------------------------------
    // done: sticky on busy falling edge, cleared by a go write
    // (v1 only: busy edge detected on the AXI clock)
    //----------------------------------------------------------------
    logic busy_q, done_ff;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            busy_q  <= 1'b0;
            done_ff <= 1'b0;
        end else begin
            busy_q <= core_busy;
            if (go_write)
                done_ff <= 1'b0;
            else if (busy_q && !core_busy)
                done_ff <= 1'b1;
        end
    end

    //----------------------------------------------------------------
    // Read channel: single outstanding, registered data
    //----------------------------------------------------------------
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[5:2])
                    A_STATUS: s_axi_rdata <= {30'b0, done_ff, core_busy};
                    default:  s_axi_rdata <= '0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
