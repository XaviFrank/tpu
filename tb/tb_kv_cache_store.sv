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
//
//   A5/A6 use a SECOND store at NUM_LAYERS=3, fed by the same write
//   chain. Three, not two or four: NUM_LAYERS must not be a power of
//   two for an out-of-range layer index to be reachable at all, and 28
//   is not a power of two. At NUM_LAYERS=4 the range check would be
//   vacuous and A6 would silently test nothing.
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
        .clk(clk), .rst_n(rst_n), .clear(clear), .w_layer(1'b0),
        .w_valid(wv), .w_idx(wi), .w_data(wd), .commit(qdone),
        .n_pos(n_pos), .full(full), .overflow(ovf),
        .rd_en(rd_en), .rd_layer(1'b0),
        .rd_pos(rd_pos), .rd_widx(rd_widx), .rd_data(rd_data));

    // Second store, three layers, driven by the SAME write chain so a
    // layer append is the real producer's timing, not a stimulus.
    logic       clear2, full2, ovf2;
    logic [1:0] lay_w, lay_r;
    logic [3:0] n_pos2;
    logic       rd_en2;
    logic [3:0] rd_pos2;
    logic [4:0] rd_widx2;
    logic [31:0] rd_data2;

    kv_cache_store #(.WORDS_PER_VEC(NW), .MAX_POS(NVEC),
                     .NUM_LAYERS(3)) cache2 (
        .clk(clk), .rst_n(rst_n), .clear(clear2), .w_layer(lay_w),
        .w_valid(wv), .w_idx(wi), .w_data(wd), .commit(qdone),
        .n_pos(n_pos2), .full(full2), .overflow(ovf2),
        .rd_en(rd_en2), .rd_layer(lay_r),
        .rd_pos(rd_pos2), .rd_widx(rd_widx2), .rd_data(rd_data2));

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

    task automatic readback2(input integer ly, input integer vec,
                             input integer wi_, output logic [31:0] got);
        begin
            @(negedge clk);
            rd_en2   = 1'b1;
            lay_r    = ly[1:0];
            rd_pos2  = vec[3:0];
            rd_widx2 = wi_[4:0];
            @(negedge clk);
            rd_en2   = 1'b0;
            @(negedge clk);
            got      = rd_data2;
        end
    endtask

    // check every word of one layer/slot against source vector `src`
    task automatic chk_layer(input integer ly, input integer vec,
                             input integer src, output integer nbad);
        integer ww;
        logic [31:0] gg;
        begin
            nbad = 0;
            for (ww = 0; ww < NW; ww = ww + 1) begin
                readback2(ly, vec, ww, gg);
                if (gg !== app_exp[src*NW + ww]) nbad = nbad + 1;
            end
        end
    endtask

    logic [31:0] g;
    integer bad, bad2;

    initial begin
        start = 0; clear = 0; rd_en = 0; rd_pos = 0; rd_widx = 0;
        clear2 = 0; rd_en2 = 0; lay_w = 0; lay_r = 0;
        rd_pos2 = 0; rd_widx2 = 0;
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

        // ---- A5: layers do not alias ------------------------------
        @(negedge clk); clear2 = 1'b1;
        @(negedge clk); clear2 = 1'b0;
        @(negedge clk);

        lay_w = 2'd0; append(0);       // vector 0 -> layer 0, slot 0
        lay_w = 2'd2; append(1);       // vector 1 -> layer 2, slot 0
        lay_w = 2'd2; append(5);       // vector 5 -> layer 2, slot 1

        chk_layer(0, 0, 0, bad);  errors = errors + bad;  bad2 = bad;
        chk_layer(2, 0, 1, bad);  errors = errors + bad;  bad2 = bad2 + bad;
        chk_layer(2, 1, 5, bad);  errors = errors + bad;  bad2 = bad2 + bad;
        if (bad2 != 0)
            $display("Error: A5 %0d words wrong across the three layer/slot reads", bad2);

        lay_w = 2'd0;
        @(negedge clk); @(negedge clk);
        if (n_pos2 !== 4'd1) begin
            errors = errors + 1;
            $display("Error: A5 layer 0 n_pos %0d expected 1", n_pos2);
        end
        lay_w = 2'd1;
        @(negedge clk); @(negedge clk);
        if (n_pos2 !== 4'd0) begin
            errors = errors + 1;
            $display("Error: A5 untouched layer 1 n_pos %0d expected 0", n_pos2);
        end
        lay_w = 2'd2;
        @(negedge clk); @(negedge clk);
        if (n_pos2 !== 4'd2) begin
            errors = errors + 1;
            $display("Error: A5 layer 2 n_pos %0d expected 2", n_pos2);
        end
        if (ovf2) begin
            errors = errors + 1;
            $display("Error: A5 overflow set during valid layer appends");
        end
        $display("  A5 three layers, %0d/%0d words exact, per-layer n_pos 1/0/2",
                 3*NW - bad2, 3*NW);

        // ---- A6: an out-of-range layer is loud, not aliased -------
        lay_w = 2'd3;                  // NUM_LAYERS is 3, so 3 is bad
        append(4);
        if (!ovf2) begin
            errors = errors + 1;
            $display("Error: A6 out-of-range layer did not set overflow");
        end
        // it must not have landed on the clamped layer 0 either
        chk_layer(0, 0, 0, bad);
        errors = errors + bad;
        if (bad != 0)
            $display("Error: A6 out-of-range append corrupted layer 0 (%0d words)", bad);
        lay_w = 2'd0;
        @(negedge clk); @(negedge clk);
        if (n_pos2 !== 4'd1) begin
            errors = errors + 1;
            $display("Error: A6 layer 0 n_pos advanced to %0d on a bad-layer commit",
                     n_pos2);
        end
        $display("  A6 layer 3 of 3: overflow %b, layer 0 intact, n_pos %0d",
                 ovf2, n_pos2);

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
