`timescale 1ns / 1ps
//====================================================================
// tb_poke_peek: the hardware smoke-test path, driven ENTIRELY through
// AXI-Lite at the synth parameterization (N=K=16). No whitebox writes
// and no hierarchical references - this is literally the sequence
// PYNQ will run, so anything it cannot do, the host cannot do either.
//
// Step 1 scope (weight buffer):
//   S1: STATUS reads 0 after reset (incl. the new poke_busy/poke_err)
//   S2: poke a complete 16-word weight tile, one 32-bit lane per AXI
//       write, polling poke_busy before every commit
//   S3: peek all 64 lanes back and compare byte for byte, proving the
//       bits reached the REAL tile_buffer (peek shares the
//       controller's read port; it is not a shadow copy)
//   S5: same for the activation buffer (buf_sel = 1)
//   S6: re-peek the weight tile afterwards - weight word v and
//       activation word v share an address and differ only in
//       buf_sel, so this catches a buf_sel bug in either direction
//   S4: poke_err stayed clear - no commit was ever dropped
//
// A buffer word is N*DATA_WIDTH = 128 bits but POKE_DATA is 32, so a
// word takes 4 lane writes; the lane-3 write commits it. POKE_ADDR
// needs no settle time before POKE_DATA: address and data cross as
// ONE payload, so there is no stale-address window to wait out.
//====================================================================
module tb_poke_peek;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128
    localparam int LANES      = WORD_W / 32;      // 4
    localparam int NVECS      = 3;                // activation vectors per run

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_POKEA  = 6'h20;
    localparam [5:0] ADDR_POKED  = 6'h24;
    localparam [5:0] ADDR_PEEKA  = 6'h28;
    localparam [5:0] ADDR_PEEKD  = 6'h2C;

    localparam int PEEK_SETTLE_ACLKS = 20;   // same budget as PERF_DATA
    localparam int POKE_POLL_MAX     = 200;

    //----------------------------------------------------------------
    // Clocks + resets (two genuinely asynchronous domains)
    //----------------------------------------------------------------
    logic aclk = 1'b0;
    logic cclk = 1'b0;
    logic aresetn, crst_n;
    int unsigned cclk_seed;
    int unsigned cclk_seed_dummy;

    always #5 aclk = ~aclk;    // 100 MHz AXI clock

    initial begin
        if (!$value$plusargs("SEED=%d", cclk_seed)) cclk_seed = 32'd1;
        // Icarus quirk: $urandom(seed) mutates its argument - display first
        $display("cclk jitter seed = %0d", cclk_seed);
        cclk_seed_dummy = $urandom(cclk_seed);
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
    end

    initial begin
        #20_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end

    //----------------------------------------------------------------
    // AXI-Lite signals
    //----------------------------------------------------------------
    logic        awvalid, awready, wvalid, wready, bvalid, bready;
    logic [5:0]  awaddr;
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic [1:0]  bresp;
    logic        arvalid, arready, rvalid, rready;
    logic [5:0]  araddr;
    logic [31:0] rdata;
    logic [1:0]  rresp;

    //----------------------------------------------------------------
    // Core-domain nets. The raw wr_* ports stay INACTIVE for the whole
    // test - poke is the only writer, exactly as on hardware.
    //----------------------------------------------------------------
    logic                        perf_clear;
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [WORD_W-1:0]           wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    int errors = 0;

    ws_tpu_top #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(16), .SHIFT_WIDTH(5), .DEBUG_CAPTURE(1)
    ) dut (
        .s_axi_aclk    (aclk),
        .s_axi_aresetn (aresetn),
        .s_axi_awvalid (awvalid),
        .s_axi_awready (awready),
        .s_axi_awaddr  (awaddr),
        .s_axi_wvalid  (wvalid),
        .s_axi_wready  (wready),
        .s_axi_wdata   (wdata),
        .s_axi_wstrb   (wstrb),
        .s_axi_bvalid  (bvalid),
        .s_axi_bready  (bready),
        .s_axi_bresp   (bresp),
        .s_axi_arvalid (arvalid),
        .s_axi_arready (arready),
        .s_axi_araddr  (araddr),
        .s_axi_rvalid  (rvalid),
        .s_axi_rready  (rready),
        .s_axi_rdata   (rdata),
        .s_axi_rresp   (rresp),
        .core_clk      (cclk),
        .core_rst_n    (crst_n),
        .wr_en_w       (wr_en_w),
        .wr_addr_w     (wr_addr_w),
        .wr_data_w     (wr_data_w),
        .wr_en_a       (wr_en_a),
        .wr_addr_a     (wr_addr_a),
        .wr_data_a     (wr_data_a),
        .perf_clear    (perf_clear),
        .sum_out       (sum_out),
        .sum_valid     (sum_valid),
        .q8_out        (q8_out),
        .q8_valid      (q8_valid)
    );

    //----------------------------------------------------------------
    // Golden weight tile: distinct bytes spanning both signs so a
    // lane-order or sign-extension bug cannot alias into a pass.
    //----------------------------------------------------------------
    int W  [16][16];
    int av [NVECS][16];

    function automatic [WORD_W-1:0] pack_avec(input int v);
        logic [WORD_W-1:0] word;
        int b;
        word = '0;
        for (int e = 0; e < 16; e++) begin
            b = av[v][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        return word;
    endfunction

    //----------------------------------------------------------------
    // AXI-Lite BFM (drive negedge, sample posedge)
    //----------------------------------------------------------------
    task automatic axi_write(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin
            awvalid = 1; awaddr = addr;
            wvalid  = 1; wdata  = data;
            bready  = 1;
        end
        fork
            begin
                do @(posedge aclk); while (awready !== 1'b1);
                @(negedge aclk) awvalid = 0;
            end
            begin
                do @(posedge aclk); while (wready !== 1'b1);
                @(negedge aclk) wvalid = 0;
            end
        join
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $error("axi_write: bresp = %b", bresp);
        end
        @(negedge aclk) bready = 0;
    endtask

    task automatic axi_read(input [5:0] addr, output [31:0] data);
        @(negedge aclk) begin arvalid = 1; araddr = addr; rready = 1; end
        do @(posedge aclk); while (arready !== 1'b1);
        @(negedge aclk) arvalid = 0;
        do @(posedge aclk); while (rvalid !== 1'b1);
        data = rdata;
        if (rresp !== 2'b00) begin
            errors++; $error("axi_read: rresp = %b", rresp);
        end
        @(negedge aclk) rready = 0;
    endtask

    //----------------------------------------------------------------
    // Poke helpers
    //----------------------------------------------------------------
    // STATUS[3] = poke_busy. bus_handshake DROPS a send issued while
    // busy, so an unpaced commit silently loses that word - the host
    // contract is to poll before every commit, and so does this TB.
    task automatic wait_poke_idle(input string name);
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[3] === 1'b0) return;
            n = n + 1;
            if (n == POKE_POLL_MAX) begin
                errors++;
                $error("[%s] poke_busy never cleared", name);
                return;
            end
        end
    endtask

    // One full buffer word = LANES lane writes; the last one commits.
    task automatic poke_word(input int sel, input int addr,
                             input [WORD_W-1:0] word);
        logic [31:0] a;
        for (int lane = 0; lane < LANES; lane++) begin
            a = '0;
            a[5:0] = 6'(addr);
            a[7:6] = 2'(lane);
            a[8]   = 1'(sel);
            axi_write(ADDR_POKEA, a);
            if (lane == LANES-1) wait_poke_idle("poke_word");
            axi_write(ADDR_POKED, word[lane*32 +: 32]);
        end
    endtask

    // Read one 32-bit element. Same round-trip contract as PERF_DATA:
    // write the address, let the crossing settle, then read.
    task automatic peek_elem(input int buf_sel, input int idx1,
                             input int idx0, output [31:0] val);
        logic [31:0] a;
        a = '0;
        a[5:0]   = 6'(idx0);
        a[9:6]   = 4'(idx1);
        a[11:10] = 2'(buf_sel);
        axi_write(ADDR_PEEKA, a);
        repeat (PEEK_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PEEKD, val);
    endtask

    // Pack weight row r into one 128-bit buffer word (manual two's
    // complement per byte - Icarus unpacked-element signedness is
    // unreliable, so golden models use int locals).
    function automatic [WORD_W-1:0] pack_row(input int r);
        logic [WORD_W-1:0] word;
        int b;
        word = '0;
        for (int e = 0; e < 16; e++) begin
            b = W[r][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        return word;
    endfunction

    //----------------------------------------------------------------
    // Golden: the untiled reference. Result convention is
    // y[v][j] = sum_i W[i][j] * a[v][i] (hosts store tiles
    // transposed), with scale_mult=1 / shift=0 / relu off, so the
    // INT8 answer is just the clamped INT32 sum - which is exactly
    // what lets the two peeked streams cross-check each other.
    //----------------------------------------------------------------
    function automatic int golden_sum(input int v, input int j);
        int y;
        y = 0;
        for (int i = 0; i < 16; i++)
            y = y + W[i][j]*av[v][i];
        return y;
    endfunction

    function automatic int clamp8(input int x);
        int r;
        r = x;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    task automatic poll_done(input string name);
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[1] === 1'b1) return;
            n = n + 1;
            if (n == POKE_POLL_MAX) begin
                errors++;
                $error("[%s] done never arrived (STATUS=%h)", name, v);
                return;
            end
        end
    endtask

    // Trigger a run over the currently-poked data and check BOTH
    // result streams against the untiled golden.
    task automatic run_and_check(input string name);
        logic [31:0] v;
        logic [31:0] v2;
        logic [31:0] cfgword;
        int bad_sum, bad_q8, got, exp;

        cfgword    = '0;
        cfgword[0] = 1'b1;                 // scale_mult = 1, shift = 0, relu off
        axi_write(ADDR_NVECS, NVECS);
        axi_write(ADDR_CFG,   cfgword);
        repeat (PEEK_SETTLE_ACLKS) @(negedge aclk);

        axi_write(ADDR_CTRL, 32'h1);
        poll_done(name);
        repeat (PEEK_SETTLE_ACLKS) @(negedge aclk);

        // INT32 accumulator bank (buf_sel 3): one element per column
        bad_sum = 0;
        for (int vec = 0; vec < NVECS; vec++)
            for (int j = 0; j < 16; j++) begin
                peek_elem(3, j, vec, v);
                got = v;                            // 32-bit two's complement
                exp = golden_sum(vec, j);
                if (got !== exp) begin
                    bad_sum = bad_sum + 1;
                    errors++;
                    $error("[%s int32] vec %0d col %0d = %0d, expected %0d",
                           name, vec, j, got, exp);
                end
            end
        if (bad_sum == 0)
            $display("PASS [%s] %0d INT32 sums peeked back exact vs the untiled golden",
                     name, NVECS*16);

        // REAL post-requant INT8 from silicon (buf_sel 2), four packed
        // columns per element. This is hardware evidence now, not a
        // host computation: the requant lanes are pipelined
        // (RQ_PIPE=2) and ws_result_capture consumes q8_out/q8_valid,
        // so these bytes came through the actual DSP multiply, the
        // round/shift and the saturate.
        bad_q8 = 0;
        for (int vec = 0; vec < NVECS; vec++)
            for (int lane = 0; lane < LANES; lane++) begin
                peek_elem(2, lane, vec, v);
                for (int k = 0; k < 4; k++) begin
                    got = $signed(v[k*8 +: 8]);
                    exp = clamp8(golden_sum(vec, lane*4 + k));
                    if (got !== exp) begin
                        bad_q8 = bad_q8 + 1;
                        errors++;
                        $error("[%s hw-int8] vec %0d col %0d = %0d, expected %0d",
                               name, vec, lane*4 + k, got, exp);
                    end
                end
            end
        if (bad_q8 == 0)
            $display("PASS [%s] %0d HW-requantized INT8 values exact vs golden (real requant path)",
                     name, NVECS*16);

        // Cross-check the two captured streams against each other.
        // This is the capability the dual capture exists for: if the
        // INT32 bank is right and the INT8 bank disagrees with a
        // host-side requant of it, the bug is in the requant lanes,
        // not the array. Checking each against golden separately would
        // not localise it.
        bad_q8 = 0;
        for (int vec = 0; vec < NVECS; vec++)
            for (int lane = 0; lane < LANES; lane++) begin
                peek_elem(2, lane, vec, v);
                for (int k = 0; k < 4; k++) begin
                    got = $signed(v[k*8 +: 8]);
                    peek_elem(3, lane*4 + k, vec, v2);
                    exp = clamp8($signed(v2));      // host requant of the HW INT32
                    if (got !== exp) begin
                        bad_q8 = bad_q8 + 1;
                        errors++;
                        $error("[%s int8-vs-int32] vec %0d col %0d: HW int8 %0d != requant(HW int32) %0d",
                               name, vec, lane*4 + k, got, exp);
                    end
                end
            end
        if (bad_q8 == 0)
            $display("PASS [%s] HW INT8 == host requant of HW INT32 on all %0d (array/requant localisable)",
                     name, NVECS*16);

        // The host-side clamp only means anything if the sums actually
        // leave INT8 range; otherwise clamping is never exercised.
        // Same degeneracy discipline as the tiling engine TB's
        // partial-vs-final check.
        bad_q8 = 0;
        for (int vec = 0; vec < NVECS; vec++)
            for (int j = 0; j < 16; j++)
                if (golden_sum(vec, j) != clamp8(golden_sum(vec, j)))
                    bad_q8 = bad_q8 + 1;
        if (bad_q8 == 0) begin
            errors++;
            $error("[%s] golden degenerate: no sum leaves INT8 range, so clamping is never exercised",
                   name);
        end else
            $display("PASS [%s] %0d of %0d sums leave INT8 range - clamping is genuinely exercised",
                     name, bad_q8, NVECS*16);
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [31:0] rd;
    logic [WORD_W-1:0] word;
    logic [31:0] expect_lane;

    initial begin
        aresetn = 0; crst_n = 0;
        awvalid = 0; awaddr = '0; wvalid = 0; wdata = '0; wstrb = 4'hF;
        bready = 0; arvalid = 0; araddr = '0; rready = 0;
        perf_clear = 0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;

        // Chosen so the INT32 sums saturate in BOTH directions (27 of
        // 48 above +127, 16 below -128): the INT8 stream must then
        // differ from the INT32 one, which is what makes capturing
        // both worth the area. S7's degeneracy check enforces this.
        for (int r = 0; r < 16; r++)
            for (int e = 0; e < 16; e++)
                W[r][e] = ((r*3 + e*5) % 15) - 7;      // -7..7
        for (int v = 0; v < NVECS; v++)
            for (int e = 0; e < 16; e++)
                av[v][e] = ((v*7 + e*3) % 14) - 3;     // -3..10

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        //--- S1: reset state, including the new status bits -----------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++;
            $error("[S1] STATUS after reset = %h, expected 0", rd);
        end else
            $display("PASS [S1] STATUS (busy/done/cfg_err/poke_busy/poke_err) reads 0 after reset");

        //--- S2: poke a complete 16-word weight tile ------------------
        for (int r = 0; r < 16; r++) begin
            word = pack_row(r);
            poke_word(0, r, word);
        end
        $display("PASS [S2] poked a full %0d-word weight tile over AXI-Lite", K);

        //--- S3: peek every lane back and compare ---------------------
        begin
            int bad;
            bad = 0;
            for (int r = 0; r < 16; r++) begin
                word = pack_row(r);
                for (int lane = 0; lane < LANES; lane++) begin
                    peek_elem(0, lane, r, rd);
                    expect_lane = word[lane*32 +: 32];
                    if (rd !== expect_lane) begin
                        bad = bad + 1;
                        errors++;
                        $error("[S3] weight word %0d lane %0d = %h, expected %h",
                               r, lane, rd, expect_lane);
                    end
                end
            end
            if (bad == 0)
                $display("PASS [S3] all %0d weight lanes peeked back byte-exact from the real buffer",
                         16*LANES);
        end

        //--- S5: poke activation vectors, peek them back --------------
        // Same mechanism, the other buffer (POKE_ADDR[8] = 1). Peeked
        // back BEFORE any go, so a later compute failure cannot be
        // confused with a load failure.
        for (int v = 0; v < NVECS; v++) begin
            word = pack_avec(v);
            poke_word(1, v, word);
        end
        begin
            int bad;
            bad = 0;
            for (int v = 0; v < NVECS; v++) begin
                word = pack_avec(v);
                for (int lane = 0; lane < LANES; lane++) begin
                    peek_elem(1, lane, v, rd);
                    expect_lane = word[lane*32 +: 32];
                    if (rd !== expect_lane) begin
                        bad = bad + 1;
                        errors++;
                        $error("[S5] activation word %0d lane %0d = %h, expected %h",
                               v, lane, rd, expect_lane);
                    end
                end
            end
            if (bad == 0)
                $display("PASS [S5] %0d activation vectors poked and peeked back byte-exact",
                         NVECS);
        end

        //--- S6: the two buffers are independent ----------------------
        // Re-peek the weight tile after the activation pokes. Weight
        // word v and activation word v share an address and differ
        // only in buf_sel, so a buf_sel bug in either the poke commit
        // or the peek read mux would show up here as the activation
        // data having landed on top of the weight tile.
        begin
            int bad;
            bad = 0;
            for (int r = 0; r < 16; r++) begin
                word = pack_row(r);
                for (int lane = 0; lane < LANES; lane++) begin
                    peek_elem(0, lane, r, rd);
                    expect_lane = word[lane*32 +: 32];
                    if (rd !== expect_lane) begin
                        bad = bad + 1;
                        errors++;
                        $error("[S6] weight word %0d lane %0d = %h after activation pokes, expected %h",
                               r, lane, rd, expect_lane);
                    end
                end
            end
            if (bad == 0)
                $display("PASS [S6] weight tile intact after activation pokes (buf_sel is honoured both ways)");
        end

        //--- S7: run it, then peek the results back -------------------
        run_and_check("S7 round 1");

        //--- S8: a second, DIFFERENT tile+vector pair -----------------
        // Proves the buffers do not retain stale state and that poke
        // overwrites correctly rather than only working from reset.
        for (int r = 0; r < 16; r++)
            for (int e = 0; e < 16; e++)
                W[r][e] = ((r*11 + e*7) % 11) - 5;      // different rule
        for (int v = 0; v < NVECS; v++)
            for (int e = 0; e < 16; e++)
                av[v][e] = ((v*5 + e*9) % 14) - 2;      // also two-sided

        for (int r = 0; r < 16; r++) begin
            word = pack_row(r);
            poke_word(0, r, word);
        end
        for (int v = 0; v < NVECS; v++) begin
            word = pack_avec(v);
            poke_word(1, v, word);
        end
        run_and_check("S8 round 2");

        //--- S4: no commit was dropped --------------------------------
        axi_read(ADDR_STATUS, rd);
        if (rd[4] !== 1'b0) begin
            errors++;
            $error("[S4] poke_err set - a commit was dropped (STATUS=%h)", rd);
        end else
            $display("PASS [S4] poke_err clear: every commit was delivered");

        //--- S9: overwrite-in-place, peeked back, BOTH buffers --------
        // Reproduces a reported hardware symptom: activation word 0
        // reading back [1,0,0,...] with element 0 stuck at the first
        // value ever poked and elements 1-15 always 0, while the
        // weight buffer behaved correctly. S8 re-pokes both buffers
        // but only verifies through results, so overwrite-in-place was
        // never directly peeked back - this closes that gap.
        begin
            int bad;
            bad = 0;

            for (int pass = 0; pass < 2; pass++) begin
                // pass 0: 1..16   pass 1: 100..115 - the second
                // differs from the first in EVERY byte
                for (int e = 0; e < 16; e++) begin
                    av[0][e] = (pass == 0) ? (e + 1) : (100 + e);
                    W[0][e]  = (pass == 0) ? (e + 1) : (100 + e);
                end

                word = pack_avec(0);
                poke_word(1, 0, word);
                for (int lane = 0; lane < LANES; lane++) begin
                    peek_elem(1, lane, 0, rd);
                    expect_lane = word[lane*32 +: 32];
                    if (rd !== expect_lane) begin
                        bad = bad + 1;
                        errors++;
                        $error("[S9 pass%0d act] word 0 lane %0d = %h, expected %h",
                               pass, lane, rd, expect_lane);
                    end
                end

                word = pack_row(0);
                poke_word(0, 0, word);
                for (int lane = 0; lane < LANES; lane++) begin
                    peek_elem(0, lane, 0, rd);
                    expect_lane = word[lane*32 +: 32];
                    if (rd !== expect_lane) begin
                        bad = bad + 1;
                        errors++;
                        $error("[S9 pass%0d wgt] word 0 lane %0d = %h, expected %h",
                               pass, lane, rd, expect_lane);
                    end
                end
            end

            if (bad == 0)
                $display("PASS [S9] overwrite-in-place peeks back exact on BOTH buffers (2 passes, every byte changed)");
        end

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
