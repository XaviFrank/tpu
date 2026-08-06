`timescale 1ns / 1ps
//====================================================================
// tb_kv_pack_int8 - the INT8 K line writer, and the round trip back
// through the EXISTING kv_unpack.
//
//   P1 packs a real layer-0 K vector and compares all 32 words.
//   P2 decodes that line with kv_unpack #(.QBITS(8), .QJL_EN(0), .HAS_NORM(0)) and
//      requires every coordinate back exactly. That is the check that
//      matters: the format was chosen so the verified decoder needs no
//      new mode, and P2 is what proves the claim rather than asserting
//      it. If the byte order or the word count were wrong, P1
//      could still pass against a golden built from the same mistake -
//      P2 could not.
//====================================================================
module tb_kv_pack_int8;

    localparam int N  = 128;
    localparam int NW = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0;
    integer i;

    logic [7:0]  x   [N];
    logic [31:0] exp [NW];
    logic [31:0] got [NW];

    initial begin
        $readmemh("tb/kvp_x.memh", x);
        $readmemh("tb/kvp_w.memh", exp);
    end

    logic               iv, pbusy, wv, pdone;
    logic signed [23:0] idata;
    logic [4:0]         wi;
    logic [31:0]        wd;

    kv_pack_int8 #(.N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(iv), .in_data(idata), .busy(pbusy),
        .w_valid(wv), .w_idx(wi), .w_data(wd), .done(pdone));

    always_ff @(posedge clk) if (wv) got[wi] <= wd;

    // decoder, fed from the packer's own output
    logic               dv;
    logic [31:0]        dd;
    logic               ubusy, uev;
    logic [N*8-1:0]     codes;
    logic [N-1:0]       signs;
    logic [29:0]        vn, rn;

    kv_unpack #(.N(N), .QBITS(8), .QJL_EN(0), .HAS_NORM(0)) u_dec (
        .clk(clk), .rst_n(rst_n), .w_valid(dv), .w_data(dd), .busy(ubusy),
        .codes(codes), .signs(signs), .vnorm(vn), .rnorm(rn),
        .entry_valid(uev));

    logic [7:0] dec;
    assign dec = codes[(N-1-i)*8 +: 8];

    initial begin
        iv = 0; idata = 0; dv = 0; dd = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== kv_pack_int8: real layer-0 K, INT8 line ===");

        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            iv    = 1'b1;
            idata = {{16{x[i][7]}}, x[i]};
        end
        @(negedge clk);
        iv = 1'b0;
        @(posedge pdone);
        repeat (2) @(negedge clk);

        for (i = 0; i < NW; i = i + 1) begin
            if (got[i] !== exp[i]) begin
                errors = errors + 1;
                if (errors <= 4)
                    $display("Error: P1 word %0d got %08h exp %08h",
                             i, got[i], exp[i]);
            end
        end
        $display("  P1 packed %0d/%0d words exact", NW - errors, NW);

        // ---- P2: straight back through the existing decoder ----
        for (i = 0; i < NW; i = i + 1) begin
            @(negedge clk);
            dv = 1'b1;
            dd = got[i];
        end
        @(negedge clk);
        dv = 1'b0;
        // entry_valid pulses on the posedge that consumes the LAST
        // word, i.e. inside the feed loop - so an edge wait placed
        // after the loop misses it. Settle instead.
        repeat (3) @(negedge clk);

        begin : chk
            integer bad;
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                #1;
                if (dec !== x[i]) begin
                    bad = bad + 1;
                    if (bad <= 4)
                        $display("Error: P2 coord %0d decoded %02h exp %02h",
                                 i, dec, x[i]);
                end
            end
            errors = errors + bad;
            $display("  P2 decoded %0d/%0d coordinates exact via kv_unpack",
                     N - bad, N);
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #500000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
