`timescale 1ns / 1ps
//====================================================================
// tb_axi_lite_ctrl v1 (spec step 2): protocol + CTRL/STATUS checks
// against a FAKE core (busy modeled in the TB), aclk 100 MHz vs
// jittered ~263 MHz cclk, asynchronous.
//   P1: STATUS reads 0 after reset; unmapped address reads 0
//   P2: go write -> exactly one core_go pulse, busy then done seen
//       through STATUS polling
//   P3: staggered AW-first and W-first writes (channel ordering),
//       wdata bit0 = 0 so no stray go
//   P4: delayed RREADY - rvalid/rdata held stable until accepted
//   P5: second go - done clears at the go write, then re-sets
//====================================================================
module tb_axi_lite_ctrl;

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;

    localparam int POLL_MAX = 50;

    logic aclk = 1'b0;
    logic aresetn;

    logic        awvalid, awready, wvalid, wready, bvalid, bready;
    logic [5:0]  awaddr;
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic [1:0]  bresp;
    logic        arvalid, arready, rvalid, rready;
    logic [5:0]  araddr;
    logic [31:0] rdata;
    logic [1:0]  rresp;

    logic core_go, core_busy;

    int errors  = 0;
    int go_seen = 0;
    int go_base = 0;

    // Core clock: ~263 MHz average with random per-half-cycle jitter,
    // so the aclk/cclk phase relationship is genuinely asynchronous
    // and sweeps through all alignments across the run (CDC bugs that
    // pass at one fixed alignment fail at another - spec section 5)
    logic cclk = 1'b0;
    logic crst_n;

    initial begin
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
    end

    axi_lite_ctrl dut (
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
        .core_go       (core_go),
        .core_busy     (core_busy)
    );

    // 100 MHz AXI clock
    always #5 aclk = ~aclk;

    //----------------------------------------------------------------
    // Fake core: count every go pulse; busy rises 2 core cycles after
    // a pulse and holds 40 (long enough that STATUS polling over AXI
    // always observes it, even once the core clock is asynchronous)
    //----------------------------------------------------------------
    always @(posedge cclk)
        if (aresetn && core_go === 1'b1) go_seen = go_seen + 1;

    initial core_busy = 1'b0;
    always @(posedge cclk) begin
        if (aresetn && core_go === 1'b1) begin
            repeat (2) @(negedge cclk);
            core_busy = 1'b1;
            repeat (40) @(negedge cclk);
            core_busy = 1'b0;
        end
    end

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

    task automatic axi_write_aw_first(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin awvalid = 1; awaddr = addr; bready = 1; end
        do @(posedge aclk); while (awready !== 1'b1);
        @(negedge aclk) awvalid = 0;
        repeat (2) @(negedge aclk);
        wvalid = 1; wdata = data;
        do @(posedge aclk); while (wready !== 1'b1);
        @(negedge aclk) wvalid = 0;
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $error("aw_first: bresp = %b", bresp);
        end
        @(negedge aclk) bready = 0;
    endtask

    task automatic axi_write_w_first(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin wvalid = 1; wdata = data; bready = 1; end
        do @(posedge aclk); while (wready !== 1'b1);
        @(negedge aclk) wvalid = 0;
        repeat (2) @(negedge aclk);
        awvalid = 1; awaddr = addr;
        do @(posedge aclk); while (awready !== 1'b1);
        @(negedge aclk) awvalid = 0;
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $error("w_first: bresp = %b", bresp);
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

    // read with rready held low: rvalid/rdata must hold stable
    task automatic axi_read_slow(input [5:0] addr, output [31:0] data);
        logic [31:0] first;
        @(negedge aclk) begin arvalid = 1; araddr = addr; rready = 0; end
        do @(posedge aclk); while (arready !== 1'b1);
        @(negedge aclk) arvalid = 0;
        do @(posedge aclk); while (rvalid !== 1'b1);
        first = rdata;
        repeat (3) @(posedge aclk);
        if (rvalid !== 1'b1 || rdata !== first) begin
            errors++;
            $error("axi_read_slow: rvalid/rdata not held (rvalid=%b)", rvalid);
        end
        @(negedge aclk) rready = 1;
        @(posedge aclk);
        data = rdata;
        @(negedge aclk) rready = 0;
    endtask

    task automatic poll_status_bit(input int bit_idx, input string name);
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[bit_idx] === 1'b1) begin
                // Early return via flag (for Icarus compatibility)
                return;
            end
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++;
                $error("poll %s: STATUS bit %0d never set", name, bit_idx);
                return;
            end
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [31:0] rd;

    initial begin
        aresetn = 0; crst_n = 0;
        awvalid = 0; awaddr = '0; wvalid = 0; wdata = '0; wstrb = 4'hF;
        bready = 0; arvalid = 0; araddr = '0; rready = 0;

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        //--- P1: STATUS and unmapped addresses read 0 after reset -----
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS reads 0 after reset");
        axi_read(6'h18, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] unmapped 0x18 = %h, expected 0", rd);
        end else $display("PASS [P1] unmapped address reads 0");

        for (int it = 0; it < 3; it++) begin
            $display("--- async iteration %0d ---", it);

            //--- P2: go -> one pulse, busy, done ---------------------------
            go_base = go_seen;
            axi_write(ADDR_CTRL, 32'h1);
            repeat (10) @(negedge aclk);
            if (go_seen - go_base !== 1) begin
                errors++;
                $error("[P2] core_go pulses = %0d, expected 1", go_seen - go_base);
            end
            poll_status_bit(0, "P2 busy");
            poll_status_bit(1, "P2 done");
            axi_read(ADDR_STATUS, rd);
            if (rd[0] !== 1'b0) begin
                errors++; $error("[P2] busy still set after done");
            end else $display("PASS [P2] go -> one pulse -> busy -> done");

            //--- P3: staggered channel ordering, no stray go ---------------
            go_base = go_seen;
            axi_write_aw_first(ADDR_CTRL, 32'h0);
            axi_write_w_first(ADDR_CTRL, 32'h0);
            repeat (10) @(negedge aclk);
            if (go_seen - go_base !== 0) begin
                errors++;
                $error("[P3] stray go pulse (delta=%0d)", go_seen - go_base);
            end else $display("PASS [P3] staggered AW/W writes, no stray go");

            //--- P4: delayed RREADY (done still set from P2) ---------------
            axi_read_slow(ADDR_STATUS, rd);
            if (rd[1] !== 1'b1) begin
                errors++; $error("[P4] STATUS.done lost during held read");
            end else $display("PASS [P4] rvalid/rdata held until rready");

            //--- P5: second go - done clears then re-sets ------------------
            go_base = go_seen;
            axi_write(ADDR_CTRL, 32'h1);
            axi_read(ADDR_STATUS, rd);
            if (rd[1] !== 1'b0) begin
                errors++; $error("[P5] done not cleared by go write");
            end
            repeat (10) @(negedge aclk);
            if (go_seen - go_base !== 1) begin
                errors++;
                $error("[P5] core_go pulses = %0d, expected 1", go_seen - go_base);
            end
            poll_status_bit(1, "P5 done again");
            $display("PASS [P5] done cleared by go and re-set");
        end

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
