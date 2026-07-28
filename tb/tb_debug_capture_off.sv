`timescale 1ns / 1ps
//====================================================================
// tb_debug_capture_off: build guard for the DEBUG_CAPTURE = 0
// configuration.
//
//   DEBUG_CAPTURE exists so the ~10Kb capture banks, the peek read
//   mux and ws_unit's debug buffer-read mux can all be reclaimed once
//   hardware bring-up is done. Nothing else in the regression builds
//   ws_tpu_top that way - every other TB takes the default of 1 - so
//   without this the disabled configuration would rot silently and
//   the reclaim path would not actually be there when wanted.
//
//   This is deliberately elaboration-only: it instantiates the top at
//   DEBUG_CAPTURE = 0 and does nothing else. Its job is to fail the
//   build, loudly, the moment an edit assumes the debug logic is
//   always present. Functional coverage of the enabled path is
//   tb_poke_peek's.
//====================================================================
module tb_debug_capture_off;

    localparam int N     = 16;
    localparam int DEPTH = 64;

    logic                        lo;
    logic [5:0]                  addr_z;
    logic [3:0]                  strb_z;
    logic [31:0]                 data_z;
    logic [N*8-1:0]              word_z;

    logic                        awready, wready, bvalid, arready, rvalid;
    logic [1:0]                  bresp, rresp;
    logic [31:0]                 rdata;
    logic signed [31:0]          sum_out   [N];
    logic                        sum_valid [N];
    logic signed [7:0]           q8_out    [N];
    logic                        q8_valid  [N];

    assign lo     = 1'b0;
    assign addr_z = '0;
    assign strb_z = '0;
    assign data_z = '0;
    assign word_z = '0;

    ws_tpu_top #(
        .N(N), .K(16), .DATA_WIDTH(8), .ACC_WIDTH(32), .DEPTH(DEPTH),
        .VEC_W(4), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5),
        .DEBUG_CAPTURE(0)
    ) dut (
        .s_axi_aclk    (lo),
        .s_axi_aresetn (lo),
        .s_axi_awvalid (lo),
        .s_axi_awready (awready),
        .s_axi_awaddr  (addr_z),
        .s_axi_wvalid  (lo),
        .s_axi_wready  (wready),
        .s_axi_wdata   (data_z),
        .s_axi_wstrb   (strb_z),
        .s_axi_bvalid  (bvalid),
        .s_axi_bready  (lo),
        .s_axi_bresp   (bresp),
        .s_axi_arvalid (lo),
        .s_axi_arready (arready),
        .s_axi_araddr  (addr_z),
        .s_axi_rvalid  (rvalid),
        .s_axi_rready  (lo),
        .s_axi_rdata   (rdata),
        .s_axi_rresp   (rresp),
        .core_clk      (lo),
        .core_rst_n    (lo),
        .wr_en_w       (lo),
        .wr_addr_w     (addr_z),
        .wr_data_w     (word_z),
        .wr_en_a       (lo),
        .wr_addr_a     (addr_z),
        .wr_data_a     (word_z),
        .perf_clear    (lo),
        .sum_out       (sum_out),
        .sum_valid     (sum_valid),
        .q8_out        (q8_out),
        .q8_valid      (q8_valid)
    );

    initial begin
        $display("PASS: ws_tpu_top elaborates with DEBUG_CAPTURE = 0");
        $display("\n=== ALL TESTS PASSED ===");
        $finish;
    end

endmodule
