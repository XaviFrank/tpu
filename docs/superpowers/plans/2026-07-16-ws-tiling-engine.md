# ws_tiling_engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend WS past single-tile scope: `ws_tiling_engine` sweeps weight tiles N-outer/K-inner over the existing `ws_quant_unit`, accumulating raw INT32 partials in a per-column buffer and requantizing once per N-tile; `ws_tpu_tiled_top` wires it under `axi_lite_ctrl` (which gains K_TILES/N_TILES registers and a STATUS cfg_err bit).

**Architecture:** The engine is a single-clock core-domain module between the control plane and the core: it consumes the outer go/busy that `axi_lite_ctrl` already provides, issues repeated single-tile go/wait-done cycles to `ws_quant_unit` (driving the `w_tile_idx`/`a_base` ports that `ws_tpu_top` ties to 0 — the flagged extension point), taps the raw `sum_out`/`sum_valid` bottom edge into N per-column accumulator banks (accumulate-on-write: overwrite at k_tile 0, add after), and after each N-tile's K-sweep drains the banks through a single third `requant` instance to a final-result stream. `ws_quant_unit`'s own INT8 stream keeps firing per-tile and is NOT authoritative during K_TILES>1 (documented host contract). `ws_tpu_top` (single-tile) stays behaviorally untouched.

**Tech Stack:** SystemVerilog (Icarus -g2012 via ./regress.sh). Existing modules: `ws_quant_unit` and CDC primitives untouched; `axi_lite_ctrl` gets a scoped, flagged extension; `requant.sv` reused as-is (third thin tap, no fork).

## Global Constraints

- `./regress.sh` after every RTL/TB change; `=== REGRESSION GREEN ===` + exit 0 (15 TBs before this plan; 17 after). Standing user consent to run it freely.
- New files added to regress.sh's explicit per-TB lists as part of creating them (so regress additions happen inside Tasks 1 and 3, not as a separate step).
- RTL style: dull portable constructs only — no whole-array assigns, no inline sized casts, no expressions in port connections; wires shared across generate blocks declared at module level.
- Reset style: control registers async reset; datapath registers (the accumulator banks, drain data pipe) NO reset.
- TB conventions: drive negedge / sample posedge, !==/===, "ALL TESTS PASSED" banner; no $signed (manual two's complement); no enum ternary assigns; never pass an automatic task-local unpacked-array element as a task output arg; $urandom's seed arg is inout in Icarus (display seed before reseeding).
- Verified WS cycle budgets the TBs may rely on (docs/ws_design_notes.md, certified N=4 and N=16): LOAD=N+2, STREAM=n_vecs+1, FLUSH=2N+1, per-tile busy=3N+4+n_vecs; ws_quant_unit's busy rises 1 cycle after go.
- Result convention: bottom of column j = sum_i w[i][j]*a[i] (y = W_loaded^T · a; hosts store weight tiles transposed). Weight-buffer word i at tile_idx*K+i = resident row i, lane j in bits [8j+7:8j].
- DSP expectations: `ws_tiling_engine`'s sequencing/accumulate logic infers 0 DSPs (32-bit adds map to carry chains); its ONE requant tap is expected to show the usual ~2 DSPs. `axi_lite_ctrl` additions stay at 0.
- Never touch vivado/. Commit at every green regression with a descriptive message; push after green commits.
- Planning decisions locked in (flag in commits/report, human may veto):
  (a) New AXI offsets: K_TILES=0x18, N_TILES=0x1C (word decode 4'h6/4'h7, next free slots).
  (b) The MAX_N_VECS runtime guard is a sticky `cfg_err` output from the engine (set on a rejected go — n_vecs invalid, tile counts zero, or capacity exceeded — cleared on the next accepted go; busy never rises on reject), surfaced as STATUS bit [2] through a new sync_2ff in axi_lite_ctrl. This is the "guard fires" hardware behavior §8 demands; a sim-only $error would not be host-visible.
  (c) `axi_lite_ctrl` gaining ports means `ws_tpu_top` needs a tie-off-only hunk (new input tied 0, new outputs to unused nets). Functionally untouched — its TBs must stay byte-identical (checked in Task 3).
  (d) Engine drain order: for v in 0..n_vecs-1, for j in 0..N-1 (v outer, j inner), one value per cycle through a 2-stage read→requant pipe. Order is part of the res_* contract.
  (e) Buffer-capacity guards fold into cfg_err: k_tiles*n_tiles ≤ DEPTH/K (weight tiles must fit the buffer — no DMA yet) and k_tiles*n_vecs ≤ DEPTH (activation chunks must fit). Data layout contract: weight tile for (n_tile, k_tile) at tile index n_tile*k_tiles + k_tile; activation chunk for k_tile at words [k_tile*n_vecs .. k_tile*n_vecs + n_vecs - 1].

## File Structure

- `rtl/ws_tiling_engine.sv` (new, Task 1; drain+requant tap added Task 2) — FSM + per-column accumulator banks + cfg_err guard; internal names `acc`, `v_cnt`, `k_cnt`, `n_cnt`, `t_idx` are TB-peek contract.
- `tb/tb_ws_tiling_engine.sv` (new, Task 1; extended Task 2) — engine + REAL ws_quant_unit at N=4, single clock (the engine is a single-clock core-domain module; CDC lives in axi_lite_ctrl and is not re-verified here).
- `rtl/axi_lite_ctrl.sv` (modify, Task 3) — K_TILES/N_TILES registers (same dirty/send/bus_handshake pattern as N_VECS), STATUS bit[2] cfg_err via sync_2ff, TILE_W param. CDC machinery otherwise untouched — review focuses on the listed signal changes only.
- `rtl/ws_tpu_top.sv` (modify, Task 3) — tie-off hunk only.
- `rtl/ws_tpu_tiled_top.sv` (new, Task 3) — axi_lite_ctrl + ws_tiling_engine + ws_quant_unit; exposes ws_tpu_top's ports plus the final-result stream (`res_*`); raw sum/q8 stay exposed with the not-authoritative host contract in the header.
- `tb/tb_ws_tpu_tiled_top.sv` (new, Task 3; K_TILES==1 case added Task 4) — N=16, async clocks, 32×32 golden over AXI.
- `regress.sh` (modify, Tasks 1 and 3).
- `docs/perf_map.md` — NOT modified (no new counters; tile_swaps now counts one per inner go, which Task 3's TB checks via the existing map).
- `reports/report-ws-tiling-engine.md` (new, Task 5, gitignored).

---

### Task 1: engine skeleton — sequencing + accumulator + cfg_err (spec commit 1)

**Files:**
- Create: `rtl/ws_tiling_engine.sv`, `tb/tb_ws_tiling_engine.sv`
- Modify: `regress.sh`

**Interfaces:**
- Consumes: `ws_quant_unit` exactly as-is (ports listed in the TB below); its busy rises 1 cycle after go and falls when the tile completes; sum_out[j]/sum_valid[j] fire continuously during the run (column j lags column 0 by j cycles, up to min(n_vecs,N) columns valid per cycle).
- Produces (stable through Task 4): `ws_tiling_engine` with the port list below MINUS the res_* stream (Task 2 adds it). Internal contract for TB peeks and Task 2: `acc[j][v]` (N banks × MAX_N_VECS of signed ACC_WIDTH), `v_cnt[j]`, `k_cnt`, `n_cnt`, `t_idx`, latched `nv_q/kt_q/nt_q`, state names E_IDLE/E_GO/E_WAITB/E_RUN/E_ADV.

- [ ] **Step 1: Write the failing test — create tb/tb_ws_tiling_engine.sv**

Complete file (T1 accumulate-on-write + T2 sequencing/monitors + T3 cfg_err; Task 2 appends T4/T5). Fails now: `ws_tiling_engine` does not exist.

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// tb_ws_tiling_engine: unit TB for the WS tiling engine (N=4, single
// clock - the engine is a core-domain module; CDC is axi_lite_ctrl's
// job and is not re-verified here). Engine + REAL ws_quant_unit.
//   T1: K_TILES=2, N_TILES=1, nv=2 - accumulate-on-write proven by
//       hierarchical peeks: after k-tile 0 the banks hold exactly the
//       partial (tile-0) sums (fresh write), after k-tile 1 exactly
//       partial0+partial1 (added, not overwritten).
//   T2: sequencing monitors across a 2x2 sweep - w_tile_idx sequence
//       0,1,2,3; a_base 0,nv,0,nv; exactly K_TILES*N_TILES inner gos.
//   T3: cfg_err guard - n_vecs > MAX_N_VECS (engine built with
//       MAX_N_VECS=4, nv=5 written), k_tiles=0, and capacity overflow
//       are each rejected: busy never rises, cfg_err sets; a valid go
//       clears it.
//   (T4/T5 - final requant stream + intermediate-not-authoritative -
//    land with the Task 2 drain tap.)
// Golden: 8x4 weight matrix (2 K-tiles of 4x4), 8-element activation
// vectors; y[j] = sum_{i<8} W[i][j]*a[i]. Host layout per the plan's
// data-layout contract: tile (n,k) at index n*k_tiles+k, activation
// chunk k at words k*nv..k*nv+nv-1.
//====================================================================
module tb_ws_tiling_engine;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int TILE_W     = 4;
    localparam int MAX_NV     = 4;    // deliberately small: T3 exceeds it

    logic clk = 1'b0;
    logic rst_n;

    // outer command
    logic              go;
    logic [VEC_W-1:0]  n_vecs;
    logic [TILE_W-1:0] k_tiles, n_tiles;
    logic              busy, cfg_err;

    // requant config (per-tensor; identity here, checked in T4)
    logic               relu_en;
    logic signed [15:0] scale_mult;
    logic [4:0]         shift_amt;

    // engine <-> core nets
    logic                        core_go, core_busy;
    logic [VEC_W-1:0]            core_n_vecs;
    logic [$clog2(DEPTH/K)-1:0]  w_tile_idx;
    logic [$clog2(DEPTH)-1:0]    a_base;

    // core raw/quant edges
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    // buffer write ports (TB is the data loader)
    logic        wr_en_w, wr_en_a;
    logic [5:0]  wr_addr_w, wr_addr_a;
    logic [31:0] wr_data_w, wr_data_a;

    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;

    int errors = 0;

    ws_tiling_engine #(
        .N(N), .K(K), .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .TILE_W(TILE_W),
        .MAX_N_VECS(MAX_NV), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) u_eng (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .n_vecs       (n_vecs),
        .k_tiles      (k_tiles),
        .n_tiles      (n_tiles),
        .busy         (busy),
        .cfg_err      (cfg_err),
        .relu_en      (relu_en),
        .scale_mult   (scale_mult),
        .shift_amt    (shift_amt),
        .core_go      (core_go),
        .core_busy    (core_busy),
        .core_n_vecs  (core_n_vecs),
        .w_tile_idx   (w_tile_idx),
        .a_base       (a_base),
        .sum_in       (sum_out),
        .sum_valid_in (sum_valid)
    );

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) u_ws (
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (core_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (core_n_vecs),
        .busy       (core_busy),
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid),
        .q8_out     (q8_out),
        .q8_valid   (q8_valid),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    // watchdog
    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end
```

Half 2 of tb/tb_ws_tiling_engine.sv:

```systemverilog
    //----------------------------------------------------------------
    // Golden: 8x4 problem for T1 (2 K-tiles x 1 N-tile), extended to
    // 8x8 for T2/T4 (2x2). W[i][gcol] with i<8, gcol<8; tile (n,k)
    // holds resident rows k*4..k*4+3, columns n*4..n*4+3, stored at
    // buffer tile index n*k_tiles + k. av[v][i], i<8.
    //----------------------------------------------------------------
    int W  [8][8];
    int av [2][8];

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    // partial sum for K-tile k only, global column gcol
    function automatic int partial(input int k, input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 4; i++)
            y = y + W[k*4 + i][gcol]*av[v][k*4 + i];
        return y;
    endfunction

    // full K=8 dot product, global column gcol
    function automatic int full_y(input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 8; i++)
            y = y + W[i][gcol]*av[v][i];
        return y;
    endfunction

    function automatic [31:0] pack4(input int e0, input int e1,
                                    input int e2, input int e3);
        int b0, b1, b2, b3;
        begin
            b0 = (e0 < 0) ? e0 + 256 : e0;
            b1 = (e1 < 0) ? e1 + 256 : e1;
            b2 = (e2 < 0) ? e2 + 256 : e2;
            b3 = (e3 < 0) ? e3 + 256 : e3;
            pack4 = {b3[7:0], b2[7:0], b1[7:0], b0[7:0]};
        end
    endfunction

    task automatic write_w(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_w = 1; wr_addr_w = addr; wr_data_w = data; end
        @(negedge clk) wr_en_w = 0;
    endtask

    task automatic write_a(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_a = 1; wr_addr_a = addr; wr_data_a = data; end
        @(negedge clk) wr_en_a = 0;
    endtask

    // load tile (n,k) for a run with kt K-tiles: buffer index n*kt+k,
    // word r = resident row r = W[k*4+r][n*4 .. n*4+3]
    task automatic load_tile(input int n, input int k, input int kt);
        for (int r = 0; r < 4; r++)
            write_w((n*kt + k)*K + r,
                    pack4(W[k*4+r][n*4+0], W[k*4+r][n*4+1],
                          W[k*4+r][n*4+2], W[k*4+r][n*4+3]));
    endtask

    // load activation chunk k for vector v at word k*nv + v
    task automatic load_avec(input int k, input int v, input int nv);
        write_a(k*nv + v, pack4(av[v][k*4+0], av[v][k*4+1],
                                av[v][k*4+2], av[v][k*4+3]));
    endtask

    //----------------------------------------------------------------
    // Monitors: inner-done counter (core_busy falling edges) and the
    // per-inner-go tile/base recorder (T2's sequencing evidence)
    //----------------------------------------------------------------
    logic core_busy_q = 1'b0;
    int inner_done_cnt = 0;

    always @(posedge clk) begin
        if (rst_n) begin
            if (core_busy_q === 1'b1 && core_busy === 1'b0)
                inner_done_cnt = inner_done_cnt + 1;
            core_busy_q <= core_busy;
        end
    end

    int go_cnt = 0;
    int seen_tidx  [8];
    int seen_abase [8];

    always @(posedge clk) begin
        if (rst_n && core_go === 1'b1) begin
            seen_tidx[go_cnt]  = w_tile_idx;
            seen_abase[go_cnt] = a_base;
            go_cnt = go_cnt + 1;
        end
    end

    task automatic check_int(input string name, input int got, input int exp);
        if (got !== exp) begin
            errors++;
            $error("[%s] got %0d, expected %0d", name, got, exp);
        end else begin
            $display("PASS [%s] = %0d", name, got);
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0; n_vecs = '0; k_tiles = '0; n_tiles = '0;
        relu_en = 0; scale_mult = 16'sd1; shift_amt = '0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        perf_clear = 0; perf_addr = '0;

        // deterministic golden data: mixed signs, partial != full for
        // many elements (T5 relies on that; T1 checks it structurally)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                W[i][j] = ((i*5 + j*3) % 19) - 9;             // -9..9
        for (int v = 0; v < 2; v++)
            for (int i = 0; i < 8; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;           // -7..7

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        //--- T1: K_TILES=2, N_TILES=1, nv=2 - accumulate-on-write -----
        // layout for kt=2: tiles at idx 0,1; chunks at words 0..1, 2..3
        load_tile(0, 0, 2);
        load_tile(0, 1, 2);
        for (int v = 0; v < 2; v++) begin
            load_avec(0, v, 2);
            load_avec(1, v, 2);
        end

        inner_done_cnt = 0; go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd1; end
        @(negedge clk) go = 0;

        // after K-tile 0 completes (first inner busy fall), the banks
        // must hold exactly the fresh-written partial-0 sums
        while (inner_done_cnt < 1) @(negedge clk);
        repeat (2) @(negedge clk);
        for (int v = 0; v < 2; v++)
            for (int j = 0; j < 4; j++)
                if (u_eng.acc[j][v] !== partial(0, v, j)) begin
                    errors++;
                    $error("[T1 k0] acc[%0d][%0d] = %0d, expected partial0 %0d",
                           j, v, u_eng.acc[j][v], partial(0, v, j));
                end
        $display("PASS [T1 k0] banks hold fresh partial-0 sums");

        // after the sweep, banks must hold partial0+partial1 (added)
        while (busy) @(negedge clk);
        @(negedge clk);
        for (int v = 0; v < 2; v++)
            for (int j = 0; j < 4; j++)
                if (u_eng.acc[j][v] !== full_y(v, j)) begin
                    errors++;
                    $error("[T1 k1] acc[%0d][%0d] = %0d, expected full %0d",
                           j, v, u_eng.acc[j][v], full_y(v, j));
                end
        $display("PASS [T1 k1] banks accumulated, not overwritten");
        check_int("T1 inner gos", go_cnt, 2);

        //--- T2: 2x2 sweep sequencing (tiles 2,3 zero-filled) ---------
        for (int t = 2; t < 4; t++)
            for (int r = 0; r < 4; r++)
                write_w(t*K + r, 32'h0);

        inner_done_cnt = 0; go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd2; end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);

        check_int("T2 inner gos", go_cnt, 4);
        check_int("T2 tidx0", seen_tidx[0], 0);
        check_int("T2 tidx1", seen_tidx[1], 1);
        check_int("T2 tidx2", seen_tidx[2], 2);
        check_int("T2 tidx3", seen_tidx[3], 3);
        check_int("T2 abase0", seen_abase[0], 0);
        check_int("T2 abase1", seen_abase[1], 2);
        check_int("T2 abase2", seen_abase[2], 0);
        check_int("T2 abase3", seen_abase[3], 2);

        //--- T3: cfg_err guard - rejects, busy never rises ------------
        // (engine built with MAX_N_VECS=4; nv=5 exceeds it)
        go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd5; k_tiles = 4'd1; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 nv>MAX] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] nv > MAX_N_VECS rejected, cfg_err set");

        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd0; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 kt=0] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] k_tiles=0 rejected");

        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd8; n_tiles = 4'd8; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 cap] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] tile capacity overflow rejected");

        // a valid go clears cfg_err and runs to completion
        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd1; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        @(negedge clk);
        if (cfg_err !== 1'b0) begin
            errors++; $error("[T3 clear] cfg_err not cleared by accepted go");
        end else $display("PASS [T3] accepted go clears cfg_err");
        while (busy) @(negedge clk);

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
```

- [ ] **Step 2: Run to verify it fails**

Run: `iverilog -g2012 -o build/t.vvp rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv rtl/perf_counters.sv rtl/ws_perf_counters.sv rtl/requant.sv rtl/ws_quant_unit.sv tb/tb_ws_tiling_engine.sv`
Expected: FAIL — `ws_tiling_engine` unknown module.

- [ ] **Step 3: Create rtl/ws_tiling_engine.sv (v1, complete file)**

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// ws_tiling_engine v1 (tiling spec step 1): sweeps weight tiles
// N-outer / K-inner over ws_quant_unit, accumulating raw INT32
// partials in per-column banks. Loop order rationale (spec section
// 1): WS has no persistent accumulator across weight-tile swaps, so
// K-sweep partials live HERE; N-outer means only one N-tile's worth
// of accumulator state ever exists.
//
//   Sequencing: one inner go / wait-done per (n_tile, k_tile) pair.
//   Weight tile (n,k) is read from buffer tile index n*k_tiles + k
//   (t_idx just increments across the sweep); activation chunk k is
//   read from a_base = k*n_vecs. The engine does NOT source tile
//   data - w_tile_idx/a_base tell the loader (TB now, DMA later)
//   which tile is in play, mirroring OS's chunk_req pattern.
//
//   Accumulator: N banks x MAX_N_VECS x ACC_WIDTH, accumulate-on-
//   write (k_tile 0 overwrites, later k-tiles add). Banks and their
//   adders are fabric (0 DSPs expected on sequencing/accumulate).
//
//   cfg_err (sticky, cleared by the next ACCEPTED go): a go with
//   n_vecs == 0, n_vecs > MAX_N_VECS, k_tiles == 0, n_tiles == 0,
//   k_tiles*n_tiles > DEPTH/K, or k_tiles*n_vecs > DEPTH is REJECTED
//   - busy never rises. Surfaced as STATUS[2] by axi_lite_ctrl.
//
//   (Step 2 of the tiling spec adds the drain: a third requant tap
//   fires once per N-tile after its K-sweep completes. Until then
//   the banks are checked hierarchically by the TB.)
//====================================================================
module ws_tiling_engine #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int ACC_WIDTH   = 32,
    parameter int OUT_WIDTH   = 8,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int TILE_W      = 4,
    parameter int MAX_N_VECS  = 16,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
)(
    input  logic clk,
    input  logic rst_n,

    // outer command (axi_lite_ctrl's core side drives this)
    input  logic              go,
    input  logic [VEC_W-1:0]  n_vecs,
    input  logic [TILE_W-1:0] k_tiles,
    input  logic [TILE_W-1:0] n_tiles,
    output logic              busy,
    output logic              cfg_err,

    // per-tensor requant config (feeds the step-2 final tap)
    input  logic                          relu_en,
    input  logic signed [SCALE_WIDTH-1:0] scale_mult,
    input  logic [SHIFT_WIDTH-1:0]        shift_amt,

    // inner command to ws_quant_unit
    output logic                          core_go,
    input  logic                          core_busy,
    output logic [VEC_W-1:0]              core_n_vecs,
    output logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    output logic [$clog2(DEPTH)-1:0]      a_base,

    // raw bottom-edge tap (column-skewed, gated by sum_valid_in)
    input  logic signed [ACC_WIDTH-1:0]   sum_in       [N],
    input  logic                          sum_valid_in [N]
);

    localparam int TW = $clog2(DEPTH/K);
    localparam int AW = $clog2(DEPTH);

    //----------------------------------------------------------------
    // Engine FSM
    //----------------------------------------------------------------
    typedef enum logic [2:0] {
        E_IDLE,
        E_GO,      // one cycle: inner go pulse
        E_WAITB,   // wait for core busy to rise (1 cycle after go)
        E_RUN,     // collect valids, wait for core busy to fall
        E_ADV      // advance k/n counters or finish
    } e_state_t;

    e_state_t state;

    // latched command + walk counters (control: async reset)
    logic [VEC_W-1:0]  nv_q;
    logic [TILE_W-1:0] kt_q, nt_q;
    logic [TILE_W-1:0] k_cnt, n_cnt;
    logic [TW-1:0]     t_idx;
    logic [AW-1:0]     a_base_r;

    // configuration validation (combinational, evaluated at go)
    logic cfg_bad;
    logic [2*TILE_W-1:0]      kt_x_nt;
    logic [TILE_W+VEC_W-1:0]  kt_x_nv;

    assign kt_x_nt = k_tiles * n_tiles;
    assign kt_x_nv = k_tiles * n_vecs;
    assign cfg_bad = (n_vecs == '0) || (n_vecs > MAX_N_VECS)
                  || (k_tiles == '0) || (n_tiles == '0)
                  || (kt_x_nt > DEPTH/K) || (kt_x_nv > DEPTH);

    logic k_first;
    assign k_first = (k_cnt == '0);

    logic collecting;
    assign collecting = (state == E_WAITB) || (state == E_RUN);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= E_IDLE;
            nv_q     <= '0;
            kt_q     <= '0;
            nt_q     <= '0;
            k_cnt    <= '0;
            n_cnt    <= '0;
            t_idx    <= '0;
            a_base_r <= '0;
            cfg_err  <= 1'b0;
        end else begin
            unique case (state)
                E_IDLE: begin
                    if (go) begin
                        if (cfg_bad) begin
                            cfg_err <= 1'b1;
                        end else begin
                            cfg_err  <= 1'b0;
                            nv_q     <= n_vecs;
                            kt_q     <= k_tiles;
                            nt_q     <= n_tiles;
                            k_cnt    <= '0;
                            n_cnt    <= '0;
                            t_idx    <= '0;
                            a_base_r <= '0;
                            state    <= E_GO;
                        end
                    end
                end

                E_GO:    state <= E_WAITB;

                E_WAITB: if (core_busy) state <= E_RUN;

                E_RUN:   if (!core_busy) state <= E_ADV;

                E_ADV: begin
                    if (k_cnt == kt_q - 1'b1) begin
                        // K-sweep complete for this N-tile
                        if (n_cnt == nt_q - 1'b1) begin
                            state <= E_IDLE;             // sweep done
                        end else begin
                            n_cnt    <= n_cnt + 1'b1;    // next N-tile
                            k_cnt    <= '0;
                            t_idx    <= t_idx + 1'b1;
                            a_base_r <= '0;
                            state    <= E_GO;
                        end
                    end else begin
                        k_cnt    <= k_cnt + 1'b1;        // next K-tile
                        t_idx    <= t_idx + 1'b1;
                        a_base_r <= a_base_r + nv_q;
                        state    <= E_GO;
                    end
                end

                default: state <= E_IDLE;                // X recovery
            endcase
        end
    end

    //----------------------------------------------------------------
    // Outputs (Moore)
    //----------------------------------------------------------------
    assign busy        = (state != E_IDLE);
    assign core_go     = (state == E_GO);
    assign core_n_vecs = nv_q;
    assign w_tile_idx  = t_idx;
    assign a_base      = a_base_r;

    //----------------------------------------------------------------
    // Accumulator banks: N columns x MAX_N_VECS vectors, accumulate-
    // on-write. acc/adders are datapath: NO reset (k_first overwrite
    // covers post-reset X). Each generate column writes only its own
    // acc[j] row and v_cnt[j] (distinct longest static prefixes).
    //----------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc [N][MAX_N_VECS];
    logic [VEC_W-1:0]            v_cnt [N];

    for (genvar j = 0; j < N; j++) begin : gen_acc
        // per-vector write index (control: async reset)
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                v_cnt[j] <= '0;
            else if (core_go)
                v_cnt[j] <= '0;
            else if (collecting && sum_valid_in[j])
                v_cnt[j] <= v_cnt[j] + 1'b1;
        end

        // bank write (datapath: no reset)
        always_ff @(posedge clk) begin
            if (collecting && sum_valid_in[j]) begin
                if (k_first)
                    acc[j][v_cnt[j]] <= sum_in[j];
                else
                    acc[j][v_cnt[j]] <= acc[j][v_cnt[j]] + sum_in[j];
            end
        end
    end

endmodule
```

- [ ] **Step 4: Add the regress.sh entry**

After the `tb_ws_tpu_n16` entry:

```bash
# --- WS tiling engine (multi-tile sweep over ws_quant_unit) ---
run_tb tb_ws_tiling_engine rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/ws_tiling_engine.sv tb/tb_ws_tiling_engine.sv
```

- [ ] **Step 5: Run regress to verify green**

Run: `bash ./regress.sh`
Expected: 16 TBs PASS (new: tb_ws_tiling_engine with T1/T2/T3 PASS lines), `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add rtl/ws_tiling_engine.sv tb/tb_ws_tiling_engine.sv regress.sh
git commit -m "ws_tiling_engine v1: N-outer/K-inner sweep + accumulate-on-write banks (tiling spec step 1)

One inner go/wait-done per (n_tile, k_tile); w_tile_idx/a_base tell
the loader which tile is in play (OS chunk_req pattern). N banks x
MAX_N_VECS accumulate raw INT32 (k0 overwrites, later k-tiles add;
banks unreset datapath). cfg_err sticky guard rejects invalid n_vecs/
tile counts/capacity - busy never rises. TB: partial-sum bank peeks
after k0 and k1, 2x2 sequencing trace, all three reject classes."
```

---

### Task 2: drain + third requant tap + host contract (spec commit 2)

**Files:**
- Modify: `rtl/ws_tiling_engine.sv`, `tb/tb_ws_tiling_engine.sv`

**Interfaces:**
- Consumes: `requant #(ACC_WIDTH, OUT_WIDTH, SCALE_WIDTH, SHIFT_WIDTH)` as-is (combinational: acc_in/relu_en/scale_mult/shift_amt -> out).
- Produces: engine gains the final-result stream, part of every later consumer's contract: `res_valid` (1-cycle pulse per value), `res_q8` (signed OUT_WIDTH), `res_vec [VEC_W-1:0]`, `res_col [$clog2(N)-1:0]`, `res_ntile [TILE_W-1:0]`. Emission order per N-tile: v outer (0..n_vecs-1), j inner (0..N-1), one value per cycle after a 2-cycle pipe. Global column = res_ntile*N + res_col.

- [ ] **Step 1: Write the failing test — extend tb/tb_ws_tiling_engine.sv with T4/T5**

Four hunks (fails now: engine has no res_* ports).

Hunk 1 — declarations, after the `logic [31:0] wr_data_w, wr_data_a;` line region (with the other engine nets):

```systemverilog
    // final-result stream (Task 2)
    logic                        res_valid;
    logic signed [OUT_WIDTH-1:0] res_q8;
    logic [VEC_W-1:0]            res_vec;
    logic [$clog2(N)-1:0]        res_col;
    logic [TILE_W-1:0]           res_ntile;
```

Hunk 2 — u_eng instance: after `.sum_valid_in (sum_valid)` (comma-terminate it):

```systemverilog
        .sum_valid_in (sum_valid),
        .res_valid    (res_valid),
        .res_q8       (res_q8),
        .res_vec      (res_vec),
        .res_col      (res_col),
        .res_ntile    (res_ntile)
```

Hunk 3 — collectors, after the T2 sequencing monitor block:

```systemverilog
    //----------------------------------------------------------------
    // Final-result collector (T4) + last-inner-run q8 capture (T5:
    // the intermediate stream requantizes PER-TILE partials and is
    // NOT the final answer - prove it, don't just document it)
    //----------------------------------------------------------------
    int res_cnt = 0;
    int r_val [32];
    int r_v   [32];
    int r_c   [32];
    int r_nt  [32];

    always @(posedge clk) begin
        if (rst_n && res_valid === 1'b1) begin
            r_val[res_cnt] = res_q8;
            r_v[res_cnt]   = res_vec;
            r_c[res_cnt]   = res_col;
            r_nt[res_cnt]  = res_ntile;
            res_cnt = res_cnt + 1;
        end
    end

    int q8_cnt  [4];
    int q8_last [4][4];   // [col][vec] of the most recent inner run

    always @(posedge clk) begin
        if (rst_n) begin
            if (core_go === 1'b1)
                for (int j = 0; j < 4; j++) q8_cnt[j] = 0;
            for (int j = 0; j < 4; j++)
                if (q8_valid[j] === 1'b1) begin
                    q8_last[j][q8_cnt[j]] = q8_out[j];
                    q8_cnt[j] = q8_cnt[j] + 1;
                end
        end
    end
```

Hunk 4 — T4/T5 body, inserted after T3's final `while (busy) @(negedge clk);` and before the summary block:

```systemverilog
        //--- T4: full 2x2 golden through the final requant stream -----
        // reload all four REAL tiles (T2 zeroed 2/3)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++)
                load_tile(n, k, 2);

        res_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd2; end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        repeat (2) @(negedge clk);

        check_int("T4 res count", res_cnt, 16);
        begin
            int idx;
            idx = 0;
            for (int nt = 0; nt < 2; nt++)
                for (int v = 0; v < 2; v++)
                    for (int j = 0; j < 4; j++) begin
                        if (r_nt[idx] !== nt || r_v[idx] !== v || r_c[idx] !== j) begin
                            errors++;
                            $error("[T4 order] idx %0d = (nt%0d v%0d c%0d), expected (nt%0d v%0d c%0d)",
                                   idx, r_nt[idx], r_v[idx], r_c[idx], nt, v, j);
                        end
                        if (r_val[idx] !== clamp8(full_y(v, nt*4 + j), 0)) begin
                            errors++;
                            $error("[T4 value] res(nt%0d v%0d c%0d) = %0d, expected %0d",
                                   nt, v, j, r_val[idx], clamp8(full_y(v, nt*4 + j), 0));
                        end
                        idx = idx + 1;
                    end
        end
        $display("PASS [T4] 2x2 sweep: 16 final results, order + values exact");

        //--- T5: the intermediate q8 stream is NOT the final answer ---
        // last inner run was tile (n=1, k=1): its q8 = requant of the
        // k=1 PARTIAL alone. Golden guarantees at least one element
        // where clamp8(partial) != clamp8(full); check the hardware
        // agrees on both sides of that difference.
        begin
            int diffs;
            diffs = 0;
            for (int v = 0; v < 2; v++)
                for (int j = 0; j < 4; j++) begin
                    if (q8_last[j][v] !== clamp8(partial(1, v, 4 + j), 0)) begin
                        errors++;
                        $error("[T5] intermediate q8[%0d][%0d] = %0d, expected per-tile partial %0d",
                               j, v, q8_last[j][v], clamp8(partial(1, v, 4 + j), 0));
                    end
                    if (clamp8(partial(1, v, 4 + j), 0) != clamp8(full_y(v, 4 + j), 0))
                        diffs = diffs + 1;
                end
            if (diffs == 0) begin
                errors++;
                $error("[T5] golden degenerate: no element distinguishes partial from full");
            end else begin
                $display("PASS [T5] intermediate stream = per-tile partials (%0d elements differ from final; res_* is the only authoritative output)", diffs);
            end
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `bash ./regress.sh`
Expected: tb_ws_tiling_engine COMPILE FAILS (res_* not ports of ws_tiling_engine).

- [ ] **Step 3: Extend rtl/ws_tiling_engine.sv (v2) — five hunks**

Hunk 1 — header: replace the final "(Step 2 of the tiling spec adds...)" paragraph with:

```systemverilog
//   HOST CONTRACT (tiling spec section 3): during tiled operation,
//   ws_quant_unit's own INT8 stream (q8_out/q8_valid) keeps firing
//   on every inner tile and requantizes PER-TILE PARTIAL sums - it
//   is NOT authoritative when K_TILES > 1 and must be ignored by
//   anything downstream. Only this engine's res_* stream carries
//   final results (per N-tile: v outer 0..n_vecs-1, j inner
//   0..N-1, one value per cycle, 2-cycle pipe latency; global
//   column = res_ntile*N + res_col). For K_TILES == 1 both streams
//   are numerically correct; res_* remains the single contract.
//   The final tap is the THIRD instance of the shared requant core
//   (no fork), fired once per N-tile after its K-sweep completes.
```

Hunk 2 — port list: after `input logic sum_valid_in [N]` (comma-terminate it) add:

```systemverilog
    input  logic                          sum_valid_in [N],

    // final results (one per (vec, col) per N-tile, after its K-sweep)
    output logic                          res_valid,
    output logic signed [OUT_WIDTH-1:0]   res_q8,
    output logic [VEC_W-1:0]              res_vec,
    output logic [$clog2(N)-1:0]          res_col,
    output logic [TILE_W-1:0]             res_ntile
```

Hunk 3 — FSM enum + drain counters. The enum gains two states:

```systemverilog
    typedef enum logic [2:0] {
        E_IDLE,
        E_GO,      // one cycle: inner go pulse
        E_WAITB,   // wait for core busy to rise (1 cycle after go)
        E_RUN,     // collect valids, wait for core busy to fall
        E_ADV,     // advance k, or enter the drain
        E_DRAIN,   // walk banks: v outer, j inner, one value/cycle
        E_FLUSH    // 2 cycles: let the read->requant pipe emit
    } e_state_t;
```

and the register declarations gain (control, async reset — add to the reset list as zeros):

```systemverilog
    logic [VEC_W-1:0]        dr_v;
    logic [$clog2(N)-1:0]    dr_j;
    logic                    flush_cnt;
```

Hunk 4 — E_ADV's K-sweep-complete branch changes destination, and two new state arms are added. Replace the whole `E_ADV` arm with:

```systemverilog
                E_ADV: begin
                    if (k_cnt == kt_q - 1'b1) begin
                        dr_v  <= '0;                     // K-sweep done:
                        dr_j  <= '0;                     // drain this N-tile
                        state <= E_DRAIN;
                    end else begin
                        k_cnt    <= k_cnt + 1'b1;        // next K-tile
                        t_idx    <= t_idx + 1'b1;
                        a_base_r <= a_base_r + nv_q;
                        state    <= E_GO;
                    end
                end

                E_DRAIN: begin
                    if (dr_j == N - 1) begin
                        dr_j <= '0;
                        if (dr_v == nv_q - 1'b1) begin
                            flush_cnt <= 1'b0;
                            state     <= E_FLUSH;
                        end else begin
                            dr_v <= dr_v + 1'b1;
                        end
                    end else begin
                        dr_j <= dr_j + 1'b1;
                    end
                end

                E_FLUSH: begin
                    flush_cnt <= 1'b1;
                    if (flush_cnt) begin
                        if (n_cnt == nt_q - 1'b1) begin
                            state <= E_IDLE;             // sweep done
                        end else begin
                            n_cnt    <= n_cnt + 1'b1;    // next N-tile
                            k_cnt    <= '0;
                            t_idx    <= t_idx + 1'b1;
                            a_base_r <= '0;
                            state    <= E_GO;
                        end
                    end
                end
```

Hunk 5 — drain pipe + the third requant tap, appended after the accumulator generate block:

```systemverilog
    //----------------------------------------------------------------
    // Drain pipe: bank read (stage 1) -> shared requant core ->
    // registered result stream (stage 2). One value per E_DRAIN
    // cycle; res_valid trails the issue by 2 cycles (E_FLUSH covers
    // the tail, so busy stays high until the last result is out).
    // Valids are control (async reset); data regs are datapath (none).
    //----------------------------------------------------------------
    logic                        dr_issue;
    assign dr_issue = (state == E_DRAIN);

    logic                        p1_valid;
    logic signed [ACC_WIDTH-1:0] p1_acc;
    logic [VEC_W-1:0]            p1_v;
    logic [$clog2(N)-1:0]        p1_j;
    logic [TILE_W-1:0]           p1_nt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) p1_valid <= 1'b0;
        else        p1_valid <= dr_issue;
    end

    always_ff @(posedge clk) begin
        p1_acc <= acc[dr_j][dr_v];
        p1_v   <= dr_v;
        p1_j   <= dr_j;
        p1_nt  <= n_cnt;
    end

    logic signed [OUT_WIDTH-1:0] rq_final;

    requant #(
        .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_rq_final (
        .acc_in     (p1_acc),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (rq_final)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) res_valid <= 1'b0;
        else        res_valid <= p1_valid;
    end

    always_ff @(posedge clk) begin
        res_q8    <= rq_final;
        res_vec   <= p1_v;
        res_col   <= p1_j;
        res_ntile <= p1_nt;
    end
```

- [ ] **Step 4: Run regress to verify green**

Run: `bash ./regress.sh`
Expected: 16 TBs PASS; tb_ws_tiling_engine shows T1-T5 PASS lines including "[T5] ... elements differ from final". `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add rtl/ws_tiling_engine.sv tb/tb_ws_tiling_engine.sv
git commit -m "ws_tiling_engine v2: final requant tap + drain stream (tiling spec step 2)

Third instance of the shared requant core (no fork) fires once per
N-tile after its K-sweep: banks walked v-outer/j-inner, one value per
cycle through a 2-stage read->requant pipe (busy covers the tail).
HOST CONTRACT documented and PROVEN in TB: ws_quant_unit's own q8
stream requantizes per-tile partials during K_TILES>1 and is not
authoritative - T5 shows hardware intermediates == per-tile partials
!= final results, while res_* matches the 2x2 golden exactly."
```

---

### Task 3: ws_tpu_tiled_top + AXI register additions (spec commit 3)

**Files:**
- Modify: `rtl/axi_lite_ctrl.sv` (scoped extension — hunks A-H below; CDC machinery untouched, review focuses on exactly these), `rtl/ws_tpu_top.sv` (tie-off hunk), `regress.sh`
- Create: `rtl/ws_tpu_tiled_top.sv`, `tb/tb_ws_tpu_tiled_top.sv`

**Interfaces:**
- Consumes: engine from Tasks 1-2 (full port list incl. res_*); axi_lite_ctrl as committed at `6d7824d` lineage (hunk anchors below match that file).
- Produces: axi_lite_ctrl v4 — new params/ports: `parameter int TILE_W = 4`; outputs `core_k_tiles [TILE_W-1:0]`, `core_n_tiles [TILE_W-1:0]`; input `core_cfg_err`. New registers K_TILES 0x18 / N_TILES 0x1C (readback supported); STATUS bit [2] = cfg_err (sync_2ff'd level). `ws_tpu_tiled_top` with ws_tpu_top's port list + `res_valid/res_q8/res_vec/res_col/res_ntile`. SIGNAL-DESTINATION CHANGES to flag for review (spec §5): in the tiled top, `axi core_go -> engine.go` (was ws.go), `engine.busy -> axi core_busy` (was ws.busy), `axi core_n_vecs -> engine.n_vecs` and the ENGINE's `core_n_vecs -> ws.n_vecs` (was direct); everything else (cfg, perf, wr ports) unchanged in kind.

- [ ] **Step 1: axi_lite_ctrl v4 — eight hunks**

A. Parameter list: `parameter int SHIFT_WIDTH = 5` gains a sibling:
```systemverilog
    parameter int SHIFT_WIDTH = 5,
    parameter int TILE_W      = 4
```

B. Core-domain ports: after `output logic [VEC_W-1:0] core_n_vecs,` add:
```systemverilog
    output logic [TILE_W-1:0]             core_k_tiles,
    output logic [TILE_W-1:0]             core_n_tiles,
    input  logic                          core_cfg_err,
```

C. Word-address localparams: after `A_PDATA = 4'h5;` add:
```systemverilog
    localparam logic [3:0] A_KTILES = 4'h6;   // 0x18
    localparam logic [3:0] A_NTILES = 4'h7;   // 0x1C
```

D. Register/handshake state: extend the declaration block (after `psel_reg`):
```systemverilog
    logic [TILE_W-1:0] ktiles_reg;
    logic [TILE_W-1:0] ntiles_reg;
```
and each of the three grouped decl lines gains the two channels:
```systemverilog
    logic nvecs_dirty, cfg_dirty, psel_dirty, ktiles_dirty, ntiles_dirty;
    logic nvecs_send,  cfg_send,  psel_send,  ktiles_send,  ntiles_send;
    logic nvecs_hs_busy, cfg_hs_busy, psel_hs_busy, ktiles_hs_busy, ntiles_hs_busy;
```

E. The write always_ff: reset list gains `ktiles_reg <= '0; ntiles_reg <= '0; ktiles_dirty <= 1'b0; ntiles_dirty <= 1'b0; ktiles_send <= 1'b0; ntiles_send <= 1'b0;`; the per-cycle send defaults gain `ktiles_send <= 1'b0; ntiles_send <= 1'b0;`; the launch section (BEFORE the commit block, same ordering rationale) gains:
```systemverilog
            if (ktiles_dirty && !ktiles_hs_busy && !ktiles_send) begin
                ktiles_send  <= 1'b1;
                ktiles_dirty <= 1'b0;
            end
            if (ntiles_dirty && !ntiles_hs_busy && !ntiles_send) begin
                ntiles_send  <= 1'b1;
                ntiles_dirty <= 1'b0;
            end
```
and the commit case gains, before `default`:
```systemverilog
                    A_KTILES: begin
                        ktiles_reg   <= wdata_q[TILE_W-1:0];
                        ktiles_dirty <= 1'b1;
                    end
                    A_NTILES: begin
                        ntiles_reg   <= wdata_q[TILE_W-1:0];
                        ntiles_dirty <= 1'b1;
                    end
```

F. CDC: after the `u_hs_psel` instance add two more bus_handshake channels (dst_data drives the outputs directly, same as psel):
```systemverilog
    logic ktiles_valid_core, ntiles_valid_core;

    bus_handshake #(.W(TILE_W)) u_hs_ktiles (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (ktiles_reg),
        .src_send  (ktiles_send),
        .src_busy  (ktiles_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_k_tiles),
        .dst_valid (ktiles_valid_core)
    );

    bus_handshake #(.W(TILE_W)) u_hs_ntiles (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (ntiles_reg),
        .src_send  (ntiles_send),
        .src_busy  (ntiles_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_n_tiles),
        .dst_valid (ntiles_valid_core)
    );
```
and next to `u_cdc_busy` a second level synchronizer:
```systemverilog
    logic cfgerr_axi;

    sync_2ff u_cdc_cfgerr (
        .clk   (s_axi_aclk),
        .rst_n (s_axi_aresetn),
        .d     (core_cfg_err),
        .q     (cfgerr_axi)
    );
```

G. Read mux: STATUS arm becomes three bits, and the two new registers read back:
```systemverilog
                    A_STATUS: s_axi_rdata <= {29'b0, cfgerr_axi, done_ff, busy_axi};
                    A_KTILES: s_axi_rdata <= {{(32-TILE_W){1'b0}}, ktiles_reg};
                    A_NTILES: s_axi_rdata <= {{(32-TILE_W){1'b0}}, ntiles_reg};
```

H. Header: the register-map comment gains the two rows and the STATUS line gains `[2] cfg_err`; append to the HOST CONTRACT block:
```systemverilog
// - When the core sits behind ws_tiling_engine (tiled top), go/busy/
//   done describe the OUTER sweep; K_TILES/N_TILES follow the same
//   settle-before-go rule as N_VECS. STATUS[2] (cfg_err, sync_2ff'd
//   level) means the last go was REJECTED by the engine (bad n_vecs/
//   tile counts/capacity): busy never rises and done never arrives -
//   but the go write already cleared done. A host whose done poll
//   times out must check STATUS[2] before assuming a hang.
```

- [ ] **Step 2: ws_tpu_top tie-off hunk (functionally untouched)**

After the `abase_zero` declarations add:
```systemverilog
    // tiling-era axi_lite_ctrl ports, unused in the single-tile top
    logic       cfg_err_zero;
    logic [3:0] unused_k_tiles;
    logic [3:0] unused_n_tiles;
    assign cfg_err_zero = 1'b0;
```
and the `u_ctrl` instance gains, after `.core_n_vecs (core_n_vecs),`:
```systemverilog
        .core_k_tiles    (unused_k_tiles),
        .core_n_tiles    (unused_n_tiles),
        .core_cfg_err    (cfg_err_zero),
```

- [ ] **Step 3: Verify the single-tile world is untouched**

Run: `bash ./regress.sh`
Expected: all 16 TBs still PASS (tb_axi_lite_ctrl and tb_ws_tpu_n16 exercise the modified axi_lite_ctrl/ws_tpu_top: STATUS bit 2 reads 0, unmapped-read expectations unchanged since 0x18/0x1C now read back written values but those TBs never write them). `=== REGRESSION GREEN ===`, exit 0. If any single-tile TB changes behavior, the extension broke compatibility — fix before proceeding.

- [ ] **Step 4: Create rtl/ws_tpu_tiled_top.sv (complete file)**

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// ws_tpu_tiled_top: chip-level top with WS tiling - axi_lite_ctrl
// -> ws_tiling_engine -> ws_quant_unit. The engine sits BETWEEN the
// control plane and the core (this is deliberately not a wrapper
// around ws_tpu_top): AXI go/busy/done describe the OUTER sweep,
// and the engine drives the w_tile_idx/a_base ports that the
// single-tile top ties to 0. ws_tpu_top remains valid and untouched
// for direct single-tile debug/use.
//
//   HOST CONTRACT (beyond axi_lite_ctrl's own header): during tiled
//   operation (K_TILES > 1), the raw sum_out/sum_valid and the
//   ws_quant_unit q8_out/q8_valid streams exposed below fire on
//   EVERY inner tile and are NOT authoritative - q8 requantizes
//   per-tile partial sums. Only the engine's res_* stream carries
//   final results. Program K_TILES/N_TILES/N_VECS/REQUANT_CFG (all
//   settle-before-go), write go, poll done; if done never arrives
//   check STATUS[2] (cfg_err - the go was rejected).
//
//   Data layout (loader contract; TB now, DMA later): weight tile
//   (n_tile, k_tile) at buffer tile index n_tile*K_TILES + k_tile;
//   activation chunk k_tile at words k_tile*n_vecs .. +n_vecs-1.
//
//   Single parameter set feeds all three instances (coupling is
//   structural); the guards below fail the build if the register-map
//   packing breaks. Pure wiring + guards: zero DSPs from this file
//   itself (u_eng carries the one final-tap requant, ~2 DSPs).
//====================================================================
module ws_tpu_tiled_top #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5,
    parameter int TILE_W      = 4,
    parameter int MAX_N_VECS  = 16
)(
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

    // core domain
    input  logic                          core_clk,
    input  logic                          core_rst_n,

    // raw buffer write ports (future DMA/AXI-HP path)
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_a,

    // perf counter clear (no AXI bit yet - see docs/perf_map.md)
    input  logic                          perf_clear,

    // per-inner-tile edges (NOT authoritative when K_TILES > 1)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N],
    output logic signed [OUT_WIDTH-1:0]   q8_out    [N],
    output logic                          q8_valid  [N],

    // FINAL results: the engine's post-accumulation requant stream
    output logic                          res_valid,
    output logic signed [OUT_WIDTH-1:0]   res_q8,
    output logic [VEC_W-1:0]              res_vec,
    output logic [$clog2(N)-1:0]          res_col,
    output logic [TILE_W-1:0]             res_ntile
);

    //----------------------------------------------------------------
    // Register-map contract guards (same class as ws_tpu_top's)
    //----------------------------------------------------------------
    if (SCALE_WIDTH + SHIFT_WIDTH + 1 > 32) begin : gen_cfg_pack_guard
        initial begin
            $error("ws_tpu_tiled_top: REQUANT_CFG packing (%0d bits) does not fit one 32-bit register",
                   SCALE_WIDTH + SHIFT_WIDTH + 1);
            $fatal(1);
        end
    end

    if (VEC_W > 32 || TILE_W > 32) begin : gen_reg_width_guard
        initial begin
            $error("ws_tpu_tiled_top: VEC_W (%0d) or TILE_W (%0d) exceeds a 32-bit register",
                   VEC_W, TILE_W);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Control plane <-> engine <-> core wiring. SIGNAL-DESTINATION
    // CHANGES vs ws_tpu_top (the review focus per the tiling spec):
    //   axi core_go    -> u_eng.go        (was ws.go)
    //   u_eng.busy     -> axi core_busy   (was ws.busy)
    //   axi core_n_vecs-> u_eng.n_vecs; u_eng.core_n_vecs -> ws.n_vecs
    //   u_eng drives ws.go / ws.w_tile_idx / ws.a_base
    // Everything else (requant cfg, perf, write ports) keeps its
    // ws_tpu_top wiring shape.
    //----------------------------------------------------------------
    logic                          outer_go, outer_busy, cfg_err;
    logic [VEC_W-1:0]              axi_n_vecs;
    logic [TILE_W-1:0]             axi_k_tiles, axi_n_tiles;
    logic                          core_relu_en;
    logic signed [SCALE_WIDTH-1:0] core_scale_mult;
    logic [SHIFT_WIDTH-1:0]        core_shift_amt;
    logic [3:0]                    core_perf_addr;
    logic [31:0]                   core_perf_data;

    logic                          inner_go, inner_busy;
    logic [VEC_W-1:0]              inner_n_vecs;
    logic [$clog2(DEPTH/K)-1:0]    w_tile_idx;
    logic [$clog2(DEPTH)-1:0]      a_base;

    axi_lite_ctrl #(
        .VEC_W(VEC_W), .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH),
        .TILE_W(TILE_W)
    ) u_ctrl (
        .s_axi_aclk      (s_axi_aclk),
        .s_axi_aresetn   (s_axi_aresetn),
        .s_axi_awvalid   (s_axi_awvalid),
        .s_axi_awready   (s_axi_awready),
        .s_axi_awaddr    (s_axi_awaddr),
        .s_axi_wvalid    (s_axi_wvalid),
        .s_axi_wready    (s_axi_wready),
        .s_axi_wdata     (s_axi_wdata),
        .s_axi_wstrb     (s_axi_wstrb),
        .s_axi_bvalid    (s_axi_bvalid),
        .s_axi_bready    (s_axi_bready),
        .s_axi_bresp     (s_axi_bresp),
        .s_axi_arvalid   (s_axi_arvalid),
        .s_axi_arready   (s_axi_arready),
        .s_axi_araddr    (s_axi_araddr),
        .s_axi_rvalid    (s_axi_rvalid),
        .s_axi_rready    (s_axi_rready),
        .s_axi_rdata     (s_axi_rdata),
        .s_axi_rresp     (s_axi_rresp),
        .core_clk        (core_clk),
        .core_rst_n      (core_rst_n),
        .core_go         (outer_go),
        .core_busy       (outer_busy),
        .core_n_vecs     (axi_n_vecs),
        .core_k_tiles    (axi_k_tiles),
        .core_n_tiles    (axi_n_tiles),
        .core_cfg_err    (cfg_err),
        .core_relu_en    (core_relu_en),
        .core_scale_mult (core_scale_mult),
        .core_shift_amt  (core_shift_amt),
        .core_perf_addr  (core_perf_addr),
        .core_perf_data  (core_perf_data)
    );

    ws_tiling_engine #(
        .N(N), .K(K), .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .TILE_W(TILE_W),
        .MAX_N_VECS(MAX_N_VECS),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_eng (
        .clk          (core_clk),
        .rst_n        (core_rst_n),
        .go           (outer_go),
        .n_vecs       (axi_n_vecs),
        .k_tiles      (axi_k_tiles),
        .n_tiles      (axi_n_tiles),
        .busy         (outer_busy),
        .cfg_err      (cfg_err),
        .relu_en      (core_relu_en),
        .scale_mult   (core_scale_mult),
        .shift_amt    (core_shift_amt),
        .core_go      (inner_go),
        .core_busy    (inner_busy),
        .core_n_vecs  (inner_n_vecs),
        .w_tile_idx   (w_tile_idx),
        .a_base       (a_base),
        .sum_in       (sum_out),
        .sum_valid_in (sum_valid),
        .res_valid    (res_valid),
        .res_q8       (res_q8),
        .res_vec      (res_vec),
        .res_col      (res_col),
        .res_ntile    (res_ntile)
    );

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_core (
        .clk        (core_clk),
        .rst_n      (core_rst_n),
        .go         (inner_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (inner_n_vecs),
        .busy       (inner_busy),
        .perf_clear (perf_clear),
        .perf_addr  (core_perf_addr),
        .perf_data  (core_perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid),
        .q8_out     (q8_out),
        .q8_valid   (q8_valid),
        .relu_en    (core_relu_en),
        .scale_mult (core_scale_mult),
        .shift_amt  (core_shift_amt)
    );

endmodule
```

- [ ] **Step 5: Create tb/tb_ws_tpu_tiled_top.sv (complete file, two halves)**

Half 1 — module, clocks, DUT, monitors, golden, loaders, BFM:

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// tb_ws_tpu_tiled_top: multi-tile golden run over AXI at the synth
// parameterization (N=K=16). A REAL 32x32 weight matrix (2x2 tiles)
// against a reference that doesn't tile at all - full_y() sums all
// 32 products directly; the hardware must match it bit-exactly
// through the engine's post-accumulation requant stream (res_*).
//   P1: STATUS (incl. bit2 cfg_err) reads 0 after reset
//   F1: 2x2 identity sweep - outer busy observed, done polled, 64
//       final results in contract order, values exact, tile_swaps=4
//   F2: 2x2 relu sweep - engine's final tap applies relu (swaps=8)
//   F3: cfg_err over AXI - K_TILES=0 go is rejected: busy never
//       rises, STATUS[2] sets, done never arrives
//   (F4 - K_TILES==1 collapse - lands with tiling spec step 4.)
// Layout per the loader contract: tile (n,k) at index n*K_TILES+k,
// activation chunk k at words k*n_vecs..+n_vecs-1.
//====================================================================
module tb_ws_tpu_tiled_top;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int TILE_W     = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_PSEL   = 6'h10;
    localparam [5:0] ADDR_PDATA  = 6'h14;
    localparam [5:0] ADDR_KTILES = 6'h18;
    localparam [5:0] ADDR_NTILES = 6'h1C;

    localparam int POLL_MAX = 200;   // outer sweeps are long
    localparam int CFG_SETTLE_ACLKS  = 20;
    localparam int PERF_SETTLE_ACLKS = 20;

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
        #10_000_000;
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
    // TB-driven core-domain nets + DUT edges
    //----------------------------------------------------------------
    logic                        perf_clear;
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [WORD_W-1:0]           wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];
    logic                        res_valid;
    logic signed [OUT_WIDTH-1:0] res_q8;
    logic [VEC_W-1:0]            res_vec;
    logic [$clog2(N)-1:0]        res_col;
    logic [TILE_W-1:0]           res_ntile;

    int errors = 0;

    ws_tpu_tiled_top #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(16), .SHIFT_WIDTH(5), .TILE_W(TILE_W),
        .MAX_N_VECS(16)
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
        .q8_valid      (q8_valid),
        .res_valid     (res_valid),
        .res_q8        (res_q8),
        .res_vec       (res_vec),
        .res_col       (res_col),
        .res_ntile     (res_ntile)
    );

    //----------------------------------------------------------------
    // Collectors: final-result stream + last-inner-run q8 (for the
    // Task-4 K_TILES==1 equivalence check)
    //----------------------------------------------------------------
    int res_cnt = 0;
    int r_val [64];
    int r_v   [64];
    int r_c   [64];
    int r_nt  [64];

    always @(posedge cclk) begin
        if (crst_n && res_valid === 1'b1) begin
            r_val[res_cnt] = res_q8;
            r_v[res_cnt]   = res_vec;
            r_c[res_cnt]   = res_col;
            r_nt[res_cnt]  = res_ntile;
            res_cnt = res_cnt + 1;
        end
    end

    int q8_cnt  [16];
    int q8_last [16][4];

    always @(posedge cclk) begin
        if (crst_n) begin
            if (dut.inner_go === 1'b1)
                for (int j = 0; j < N; j++) q8_cnt[j] = 0;
            for (int j = 0; j < N; j++)
                if (q8_valid[j] === 1'b1) begin
                    q8_last[j][q8_cnt[j]] = q8_out[j];
                    q8_cnt[j] = q8_cnt[j] + 1;
                end
        end
    end

    //----------------------------------------------------------------
    // Golden: 32x32, reference does NOT tile (sums all 32 directly)
    //----------------------------------------------------------------
    int W  [32][32];
    int av [2][32];

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    function automatic int full_y(input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 32; i++)
            y = y + W[i][gcol]*av[v][i];
        return y;
    endfunction

    //----------------------------------------------------------------
    // Loaders (128-bit words, manual two's complement per byte)
    //----------------------------------------------------------------
    task automatic write_w_word(input int addr, input [WORD_W-1:0] word);
        @(negedge cclk) begin wr_en_w = 1; wr_addr_w = 6'(addr); wr_data_w = word; end
        @(negedge cclk) wr_en_w = 0;
    endtask

    task automatic write_a_word(input int addr, input [WORD_W-1:0] word);
        @(negedge cclk) begin wr_en_a = 1; wr_addr_a = 6'(addr); wr_data_a = word; end
        @(negedge cclk) wr_en_a = 0;
    endtask

    // tile (n,k) for a kt-wide sweep: index n*kt+k, word r = resident
    // row r = W[k*16+r][n*16 .. n*16+15]
    task automatic load_tile(input int n, input int k, input int kt);
        logic [WORD_W-1:0] word;
        int b;
        for (int r = 0; r < 16; r++) begin
            for (int e = 0; e < 16; e++) begin
                b = W[k*16 + r][n*16 + e];
                if (b < 0) b = b + 256;
                word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
            end
            write_w_word((n*kt + k)*K + r, word);
        end
    endtask

    // activation chunk k for vector v at word k*nv + v
    task automatic load_avec(input int k, input int v, input int nv);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < 16; e++) begin
            b = av[v][k*16 + e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        write_a_word(k*nv + v, word);
    endtask
```

Half 2 — BFM, sweep helper, main sequence:

```systemverilog
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

    task automatic poll_status_bit(input int bit_idx, input string name);
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[bit_idx] === 1'b1) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++;
                $error("poll %s: STATUS bit %0d never set", name, bit_idx);
                return;
            end
        end
    endtask

    int exp_swaps = 0;

    task automatic perf_check_swaps(input string name);
        logic [31:0] v;
        axi_write(ADDR_PSEL, {28'b0, 4'd7});
        repeat (PERF_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PDATA, v);
        if (v !== exp_swaps) begin
            errors++;
            $error("[%s] tile_swaps = %0d, expected %0d", name, v, exp_swaps);
        end else begin
            $display("PASS [%s] tile_swaps = %0d (one per inner go)", name, v);
        end
    endtask

    //----------------------------------------------------------------
    // Program registers and run one full sweep (checks done by caller)
    //----------------------------------------------------------------
    task automatic run_sweep(input int nv, input int kt, input int nt,
                             input int relu, input string name);
        logic [31:0] st;
        logic [31:0] cfgword;

        cfgword = '0;
        cfgword[0] = 1'b1;                 // scale_mult = 1, shift = 0
        if (relu != 0) cfgword[21] = 1'b1;

        axi_write(ADDR_NVECS,  nv);
        axi_write(ADDR_CFG,    cfgword);
        axi_write(ADDR_KTILES, kt);
        axi_write(ADDR_NTILES, nt);
        axi_read(ADDR_KTILES, st);
        if (st !== kt) begin
            errors++; $error("[%s] K_TILES readback = %0d, wrote %0d", name, st, kt);
        end
        axi_read(ADDR_NTILES, st);
        if (st !== nt) begin
            errors++; $error("[%s] N_TILES readback = %0d, wrote %0d", name, st, nt);
        end
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);

        res_cnt = 0;
        axi_write(ADDR_CTRL, 32'h1);

        // outer busy holds for the whole sweep (hundreds of cclk):
        // trivially observable a few aclk after the commit
        repeat (2) @(negedge aclk);
        axi_read(ADDR_STATUS, st);
        if (st[0] !== 1'b1) begin
            errors++; $error("[%s] outer STATUS.busy not observed", name);
        end else $display("PASS [%s] outer STATUS.busy observed", name);

        poll_status_bit(1, name);
        repeat (10) @(negedge aclk);
        exp_swaps = exp_swaps + kt*nt;
    endtask

    // check the collected res stream against a golden K-total
    task automatic check_res(input int nv, input int nt, input int ktotal,
                             input int relu, input string name);
        int idx, y;
        if (res_cnt !== nt*nv*16) begin
            errors++;
            $error("[%s] res count = %0d, expected %0d", name, res_cnt, nt*nv*16);
        end else begin
            idx = 0;
            for (int t = 0; t < nt; t++)
                for (int v = 0; v < nv; v++)
                    for (int j = 0; j < 16; j++) begin
                        if (r_nt[idx] !== t || r_v[idx] !== v || r_c[idx] !== j) begin
                            errors++;
                            $error("[%s order] idx %0d = (nt%0d v%0d c%0d), expected (nt%0d v%0d c%0d)",
                                   name, idx, r_nt[idx], r_v[idx], r_c[idx], t, v, j);
                        end
                        y = 0;
                        for (int i = 0; i < ktotal; i++)
                            y = y + W[i][t*16 + j]*av[v][i];
                        if (r_val[idx] !== clamp8(y, relu)) begin
                            errors++;
                            $error("[%s value] res(nt%0d v%0d c%0d) = %0d, expected %0d (int32 %0d)",
                                   name, t, v, j, r_val[idx], clamp8(y, relu), y);
                        end
                        idx = idx + 1;
                    end
            $display("PASS [%s] %0d final results, order + values exact vs untiled reference", name, res_cnt);
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
        perf_clear = 0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;

        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                W[i][j] = ((i*5 + j*3) % 19) - 9;             // -9..9
        for (int v = 0; v < 2; v++)
            for (int i = 0; i < 32; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;           // -7..7

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        // preload the full 2x2 layout: 4 tiles (all 64 weight words)
        // + 2 chunks x 2 vectors (activation words 0..3)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++)
                load_tile(n, k, 2);
        for (int k = 0; k < 2; k++)
            for (int v = 0; v < 2; v++)
                load_avec(k, v, 2);

        //--- P1: reset state (incl. the new cfg_err bit) ---------------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS (busy/done/cfg_err) reads 0 after reset");

        //--- F1: 2x2 identity sweep, 32x32 golden ----------------------
        run_sweep(2, 2, 2, 0, "F1 2x2 identity");
        check_res(2, 2, 32, 0, "F1 2x2 identity");
        perf_check_swaps("F1");

        //--- F2: 2x2 relu sweep (final tap applies relu) ---------------
        run_sweep(2, 2, 2, 1, "F2 2x2 relu");
        check_res(2, 2, 32, 1, "F2 2x2 relu");
        perf_check_swaps("F2");

        //--- F3: cfg_err over AXI - K_TILES=0 go is rejected -----------
        axi_write(ADDR_KTILES, 32'h0);
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);
        axi_write(ADDR_CTRL, 32'h1);
        repeat (30) @(negedge aclk);
        axi_read(ADDR_STATUS, rd);
        if (rd[2] !== 1'b1 || rd[0] !== 1'b0) begin
            errors++;
            $error("[F3] STATUS = %h, expected cfg_err=1 busy=0", rd);
        end else $display("PASS [F3] K_TILES=0 rejected: cfg_err set, busy never rose");

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
```

- [ ] **Step 6: regress.sh — add the tiled-top entry**

After the `tb_ws_tiling_engine` entry:

```bash
run_tb tb_ws_tpu_tiled_top rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_tiling_engine.sv \
                         rtl/ws_tpu_tiled_top.sv tb/tb_ws_tpu_tiled_top.sv
```

- [ ] **Step 7: Run regress to verify green**

Run: `bash ./regress.sh`
Expected: 17 TBs PASS (P1/F1/F2/F3 lines in the new one), `=== REGRESSION GREEN ===`, exit 0. Also run `vvp build/tb_ws_tpu_tiled_top.vvp +SEED=777` — must end `ALL TESTS PASSED` (phase-trajectory sweep).

- [ ] **Step 8: Commit**

```bash
git add rtl/axi_lite_ctrl.sv rtl/ws_tpu_top.sv rtl/ws_tpu_tiled_top.sv tb/tb_ws_tpu_tiled_top.sv regress.sh
git commit -m "ws_tpu_tiled_top + axi_lite_ctrl v4: tiled sweeps over AXI (tiling spec step 3)

axi_lite_ctrl gains K_TILES (0x18) / N_TILES (0x1C) via the existing
bus_handshake pattern and STATUS[2] = cfg_err (sync_2ff level; a
rejected go never raises busy and never delivers done - documented in
the HOST CONTRACT). Signal-destination changes, for review focus: go/
busy now terminate at ws_tiling_engine (outer sweep), n_vecs routes
through it; CDC machinery itself untouched. ws_tpu_top: tie-off-only
hunk, single-tile TBs unchanged and green. TB: 32x32 (2x2 tiles at
N=16) golden over AXI vs an untiled reference, relu sweep, cfg_err
over AXI, tile_swaps exact."
```

---

### Task 4: K_TILES == 1 collapses to the single-tile path (spec commit 4)

**Files:**
- Modify: `tb/tb_ws_tpu_tiled_top.sv`

**Interfaces:** consumes everything above; adds no RTL. The check: with K_TILES=1 the engine's res_* and ws_quant_unit's own q8 stream must AGREE (both are numerically correct for a single-tile K-sweep) and match the untiled 16-row reference — proving the engine changes nothing for the case that already worked.

- [ ] **Step 1: Add F4 to the main sequence (before the summary block)**

```systemverilog
        //--- F4: K_TILES==1 collapse - identical to single-tile path ---
        // tile (0,0) already sits at index 0 and chunk 0 at words 0..1
        // (same layout for kt=1); accepted go must also clear cfg_err
        run_sweep(2, 1, 1, 0, "F4 kt=1 collapse");
        check_res(2, 1, 16, 0, "F4 kt=1 collapse");
        perf_check_swaps("F4");

        axi_read(ADDR_STATUS, rd);
        if (rd[2] !== 1'b0) begin
            errors++; $error("[F4] cfg_err not cleared by accepted go");
        end else $display("PASS [F4] accepted go cleared cfg_err");

        // the inner q8 stream (captured from the single inner run) and
        // the engine's final stream must agree elementwise: kt=1 means
        // both requantize the same full sums
        begin
            int mismatches;
            mismatches = 0;
            for (int v = 0; v < 2; v++)
                for (int j = 0; j < 16; j++) begin
                    if (q8_cnt[j] !== 2) begin
                        errors++;
                        $error("[F4 q8] column %0d: %0d inner results, expected 2", j, q8_cnt[j]);
                    end else if (q8_last[j][v] !== r_val[v*16 + j]) begin
                        mismatches = mismatches + 1;
                        errors++;
                        $error("[F4 q8] col %0d vec %0d: inner q8 %0d != final res %0d",
                               j, v, q8_last[j][v], r_val[v*16 + j]);
                    end
                end
            if (mismatches == 0)
                $display("PASS [F4] kt=1: inner q8 stream == engine res stream (collapse proven)");
        end
```

Also update the TB header comment: replace the `(F4 - K_TILES==1 collapse - lands with tiling spec step 4.)` line with `F4: K_TILES==1 collapse - res_* == inner q8 == untiled 16-row reference; accepted go clears cfg_err.`

- [ ] **Step 2: Run regress to verify green**

Run: `bash ./regress.sh`
Expected: 17 TBs PASS with F4 lines present. `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add tb/tb_ws_tpu_tiled_top.sv
git commit -m "tb_ws_tpu_tiled_top F4: K_TILES==1 collapses to the single-tile path (tiling spec step 4)

With one K-tile the engine's post-accumulation requant and
ws_quant_unit's own q8 stream requantize the same sums: proven equal
elementwise and equal to the untiled 16-row reference; accepted go
clears cfg_err."
```

---

### Task 5: report, final green, push

**Files:**
- Create: `reports/report-ws-tiling-engine.md` (gitignored — not committed)

- [ ] **Step 1: Final full regression**

Run: `bash ./regress.sh` — 17 TBs PASS, `=== REGRESSION GREEN ===`, exit 0. Also one alternate-seed run of each async TB (`vvp build/tb_ws_tpu_tiled_top.vvp +SEED=31415`).

- [ ] **Step 2: Write the report**

Per the standing workflow (report per spec, gitignored). Cover: per-spec-step commits; the loop-order/accumulator design as built; the res_* stream contract + drain order; deviations locked in the plan header (AXI offsets chosen, cfg_err STATUS[2] mechanism, ws_tpu_top tie-off hunk, capacity guards); the proven host contract (T5/F4 evidence — intermediates == per-tile partials != finals; kt=1 collapse); Vivado handoff: `ws_tpu_tiled_top` is an ALTERNATIVE synth top (ws_tpu_top stays valid); expected deltas vs the single-tile run — engine sequencing/accumulate 0 DSP, +2 DSP for the final requant tap, accumulator banks N*MAX_N_VECS*32b in fabric/LUTRAM; same clocks/XDC apply unchanged; recheck HOLD as always.

- [ ] **Step 3: Push**

```bash
git push origin main
git log origin/main -5 --oneline
```

## Execution notes

- Tasks strictly sequential (each is a spec commit gate). Task 3's Step 3 (single-tile TBs unchanged) is a hard gate: if tb_axi_lite_ctrl or tb_ws_tpu_n16 change behavior, the axi_lite_ctrl extension broke compatibility.
- tb_axi_lite_ctrl's P1 reads 0x18 expecting 0 — that address is now the K_TILES readback, which RESETS to 0, so the check still passes unchanged. Do not "fix" that TB; the value contract holds.
- The engine TB peeks u_eng.acc/u_eng internals hierarchically and the tiled-top TB references dut.inner_go — internal names are contract (File Structure section); renaming them breaks TBs by design.
- If a timing assumption fails under jitter (e.g. drain tail vs busy), debug the engine's E_FLUSH accounting first; never widen a TB tolerance.
