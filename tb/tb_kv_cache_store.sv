`timescale 1ns / 1ps
//====================================================================
// tb_kv_cache_store - append eight real K vectors through the full
// write path and read every word back.
//
//   The chain is the real one: rotate -> quantize -> append. Nothing
//   is injected mid-path, so an addressing bug has to survive the
//   actual producer's timing rather than a convenient stimulus.
//
//   THE EIGHT VECTORS ARE DISTINCT (token 0's eight KV heads, checked
//   distinct by the generator). Appending one vector eight times would
//   pass even if every slot aliased to the same address, which is the
//   single most likely addressing bug and the one a lazier stimulus
//   would hide.
//
//   MAX_POS is set to exactly 8 so the capacity boundary is reachable:
//   A1 fills the cache, A3 attempts a ninth append and requires BOTH
//   that overflow asserts AND that slot 7 still reads back correctly.
//   An overflow that quietly wrapped would corrupt the oldest entries
//   while every status bit still looked healthy.
//====================================================================
module tb_kv_cache_store;

    localparam int N    = 128;
    localparam int NW   = 14;      // K format: 8 idx + norm + 4 sign + rnorm
    localparam int NVEC = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0;

    logic [7:0]  app_x   [NVEC*N];
    logic [31:0] app_exp [NVEC*NW];

    initial begin
        $readmemh("tb/kv_app_x.memh",   app_x);
        $readmemh("tb/kv_app_exp.memh", app_exp);
    end

    //----------------------------------------------------------------
    // rotate -> quantize -> store
    //----------------------------------------------------------------
    logic [7:0]         xbuf [N];
    logic               start, rbusy, rd_en_r, rdone;
    logic [6:0]         raddr;
    logic signed [7:0]  rdata;
    logic               yv;
    logic [6:0]         yi;
    logic signed [23:0] yd;
    logic               qbusy, wv, qdone;
    logic [4:0]         wi;
    logic [31:0]        wd;

    logic               clear, full, ovf;
    logic [3:0]         n_pos;
    logic               rd_en;
    logic [3:0]         rd_pos;
    logic [4:0]         rd_widx;
    logic [31:0]        rd_data;

    always_ff @(posedge clk) rdata <= xbuf[raddr];

    kv_rotate #(.N(N), .ROT_MODE(0)) rot (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(rbusy),
        .x_rd_en(rd_en_r), .x_rd_addr(raddr), .x_rd_data(rdata),
        .y_valid(yv), .y_idx(yi), .y_out(yd), .done(rdone));

    kv_quantize #(.N(N), .QBITS(2), .QJL_EN(1),
                  .CB_PATH("rtl/kv_cb_m0_b2.memh")) qnt (
        .clk(clk), .rst_n(rst_n), .in_valid(yv), .in_data(yd),
        .busy(qbusy), .w_valid(wv), .w_idx(wi), .w_data(wd), .done(qdone));

    kv_cache_store #(.WORDS_PER_VEC(NW), .MAX_POS(NVEC)) cache (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .w_valid(wv), .w_idx(wi), .w_data(wd), .commit(qdone),
        .n_pos(n_pos), .full(full), .overflow(ovf),
        .rd_en(rd_en), .rd_pos(rd_pos), .rd_widx(rd_widx), .rd_data(rd_data));

    //----------------------------------------------------------------
    integer v, w;

    task automatic append(input integer which);
        integer j;
        begin
            for (j = 0; j < N; j = j + 1)
                xbuf[j] = app_x[which*N + j];
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            // commit lands on the posedge AFTER qdone rises, so two
            // negedges are needed before pos/full/overflow are stable.
            // One posedge samples in the same delta as the update and
            // reads the pre-commit value.
            @(posedge qdone);
            repeat (2) @(negedge clk);
        end
    endtask

    task automatic readback(input integer vec, input integer wi_,
                            output logic [31:0] got);
        begin
            @(negedge clk);
            rd_en   = 1'b1;
            rd_pos  = vec[3:0];
            rd_widx = wi_[4:0];
            @(negedge clk);
            rd_en   = 1'b0;
            @(negedge clk);
            got     = rd_data;
        end
    endtask

    logic [31:0] g;
    integer bad;

    initial begin
        start = 0; clear = 0; rd_en = 0; rd_pos = 0; rd_widx = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== kv_cache_store: append %0d real K vectors ===", NVEC);

        // ---- A1: append all eight ---------------------------------
        for (v = 0; v < NVEC; v = v + 1) append(v);
        if (n_pos !== NVEC[3:0]) begin
            errors = errors + 1;
            $display("Error: A1 n_pos %0d expected %0d", n_pos, NVEC);
        end
        if (full !== 1'b1) begin
            errors = errors + 1;
            $display("Error: A1 full not asserted after %0d appends", NVEC);
        end
        if (ovf !== 1'b0) begin
            errors = errors + 1;
            $display("Error: A1 overflow asserted early");
        end
        $display("  A1 appended %0d vectors, n_pos %0d, full %b, overflow %b",
                 NVEC, n_pos, full, ovf);

        // ---- A2: every word of every slot -------------------------
        bad = 0;
        for (v = 0; v < NVEC; v = v + 1) begin
            for (w = 0; w < NW; w = w + 1) begin
                readback(v, w, g);
                if (g !== app_exp[v*NW + w]) begin
                    bad = bad + 1;
                    if (bad <= 5)
                        $display("Error: A2 slot %0d word %0d got %08h exp %08h",
                                 v, w, g, app_exp[v*NW + w]);
                end
            end
        end
        errors = errors + bad;
        $display("  A2 read back %0d/%0d words exact",
                 NVEC*NW - bad, NVEC*NW);

        // ---- A3: overflow is loud, and does not corrupt -----------
        append(0);
        if (ovf !== 1'b1) begin
            errors = errors + 1;
            $display("Error: A3 ninth append did not set overflow");
        end
        if (n_pos !== NVEC[3:0]) begin
            errors = errors + 1;
            $display("Error: A3 n_pos advanced past capacity to %0d", n_pos);
        end
        bad = 0;
        for (w = 0; w < NW; w = w + 1) begin
            readback(NVEC-1, w, g);
            if (g !== app_exp[(NVEC-1)*NW + w]) bad = bad + 1;
        end
        if (bad != 0) begin
            errors = errors + bad;
            $display("Error: A3 overflow corrupted slot %0d (%0d words)",
                     NVEC-1, bad);
        end
        $display("  A3 ninth append: overflow %b, n_pos %0d, last slot intact",
                 ovf, n_pos);

        // ---- A4: clear restores an empty, writable cache ----------
        @(negedge clk); clear = 1'b1;
        @(negedge clk); clear = 1'b0;
        if (n_pos !== 4'd0 || full !== 1'b0 || ovf !== 1'b0) begin
            errors = errors + 1;
            $display("Error: A4 clear left n_pos %0d full %b overflow %b",
                     n_pos, full, ovf);
        end
        append(3);
        bad = 0;
        for (w = 0; w < NW; w = w + 1) begin
            readback(0, w, g);
            if (g !== app_exp[3*NW + w]) bad = bad + 1;
        end
        errors = errors + bad;
        $display("  A4 after clear: n_pos %0d, re-append lands at slot 0, %0d/%0d exact",
                 n_pos, NW - bad, NW);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
