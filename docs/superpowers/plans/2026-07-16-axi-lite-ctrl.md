# AXI-Lite Control Wrapper (WS, single-tile) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AXI-Lite control/status register file for the WS unit (go/busy/done, N_VECS, REQUANT_CFG, perf readback) across a real two-clock CDC boundary, plus the prerequisite WS perf_counters port.

**Architecture:** A new `ws_perf_counters` mapping wrapper reuses the existing `perf_counters` hardware inside `ws_unit`. A standalone `axi_lite_ctrl` module owns the AXI-Lite subordinate FSM (PS clock domain) and a core-clock-domain side; three small CDC primitives (`sync_2ff`, `pulse_sync`, `bus_handshake`) carry busy, go/done, and multi-bit values between them. `axi_lite_ctrl` does NOT instantiate the WS stack — the integration TB wires them together (control-plane module stays synthesizable/checkable alone).

**Tech Stack:** SystemVerilog (Icarus `-g2012` via ./regress.sh), existing WS RTL untouched except additive ports.

## Global Constraints

- Run `./regress.sh` after every RTL or TB change; green banner (`=== REGRESSION GREEN ===`) + exit 0 required before a task is done. CLAUDE.md marks human confirmation as vital (rule "pending human change"): confirm with the human before the first regress run of a session if they are present; otherwise the standing rule is to run it.
- New files MUST be added to regress.sh's per-TB explicit file lists as part of creating them.
- RTL style: dull portable constructs only — no whole-array assigns, no inline sized casts, no expressions in port connections; wires shared across generate blocks declared at module level.
- Reset style: control registers async reset (`always_ff @(posedge clk or negedge rst_n)`), datapath registers none. All registers in this plan are control-plane: async reset.
- TB conventions: drive negedge / sample posedge, `!==`/`===` comparisons, self-checking with an `ALL TESTS PASSED` banner (regress greps for that substring).
- Icarus quirks (from project memory): no `$signed` in TBs (manual two's complement packing), no enum ternary assigns (use if/else), never pass an automatic task-local unpacked-array element as a task output arg (vvp segfault — use scalar locals or module-scope scalars).
- Semgrep guardian may block the Edit tool: if Edit fails, Read the file, apply the hunk manually, and Write the full file.
- Never touch `vivado/` (or `constraints/` — synthesis and XDC are the human's).
- Commit at every green regression with a descriptive message; push to origin after green commits.
- Spec deviations locked in during planning (flag in commit messages/report): (a) added a `flush` counter — the WS FSM has a fourth state `W_FLUSH` the spec's proposed set missed; (b) `w_tile_idx`/`a_base` have no register in the spec's map — tie to 0 in the integration TB, single-tile scope; (c) CDC primitives live in three new small RTL files not listed in the spec's §4 file plan.
- Control-plane module: if DSPs appear for `axi_lite_ctrl` in synth, that is a bug signal (human checks; note in report).

## File Structure

- `rtl/ws_perf_counters.sv` (new, Task 1) — WS category mapping wrapper around `perf_counters` (unchanged). WS counter map contract: `0=total, 1=load, 2=stream, 3=flush, 4=unused(0), 5=idle, 6=unused(0), 7=tile_swaps`. Keeps `total=0`, `idle=5`, tile-count=7 at the same indices as the OS map in docs/perf_map.md (host-code friendly).
- `rtl/ws_controller.sv` (modify, Task 1) — add Moore flag outputs `in_stream`, `in_flush`, `in_idle` (in_load is the existing `load_mode`; the tile-swap strobe is the existing `w_capture`).
- `rtl/ws_unit.sv` (modify, Task 1) — new ports `perf_clear`, `perf_addr[3:0]`, `perf_data[31:0]`; instantiate `ws_perf_counters`.
- `rtl/ws_quant_unit.sv` (modify, Task 1) — pass the three perf ports through.
- `tb/tb_ws_controller.sv` (modify, Task 1) — T4 perf-counter checks.
- `tb/tb_ws_quant_unit.sv`, `tb/tb_ws_n16.sv` (modify, Task 1) — tie off new ports.
- `rtl/axi_lite_ctrl.sv` (new, Task 2; reworked Tasks 3 and 4) — AXI-Lite subordinate + register file + core-side ports.
- `tb/tb_axi_lite_ctrl.sv` (new, Task 2; reworked Tasks 3 and 4) — protocol TB with fake core, then async clocks, then full integration with `ws_quant_unit`.
- `rtl/sync_2ff.sv`, `rtl/pulse_sync.sv` (new, Task 3) — level and pulse CDC primitives.
- `rtl/bus_handshake.sv` (new, Task 4) — multi-bit value req/ack toggle handshake.
- `regress.sh` (modify, Tasks 1–4) — extend three existing WS lists; add `tb_axi_lite_ctrl` entry.
- `docs/perf_map.md` (modify, Task 5) — append the WS counter map contract.
- `reports/report-axi-lite-ctrl.md` (new, Task 5, gitignored) — progress report incl. Vivado handoff asks.

---

### Task 1: WS perf_counters port (§1 — spec commit step 1)

**Files:**
- Create: `rtl/ws_perf_counters.sv`
- Modify: `rtl/ws_controller.sv`, `rtl/ws_unit.sv`, `rtl/ws_quant_unit.sv`
- Modify: `tb/tb_ws_controller.sv`, `tb/tb_ws_quant_unit.sv`, `tb/tb_ws_n16.sv`, `regress.sh`

**Interfaces:**
- Consumes: existing `perf_counters #(NCTR=8, CW=32)` (indexed registered read: `perf_data` valid 1 cycle after `perf_addr`; out-of-range reads 0; saturating; `perf_clear` zeroes).
- Produces: `ws_unit` and `ws_quant_unit` each gain ports `input logic perf_clear`, `input logic [3:0] perf_addr`, `output logic [31:0] perf_data` with the WS counter map above. Cycle budgets (already TB-certified): load = N+2, stream = n_vecs+1, flush = 2N+1, one `tile_swaps` count per accepted go. Task 4's TB relies on exactly these.

- [ ] **Step 1: Write the failing test — extend tb/tb_ws_controller.sv with perf ports + T4**

Four hunks (if Edit is blocked, Read + Write full file):

Hunk 1 — declarations, after `logic signed [ACC_WIDTH-1:0] sum_out [N];` / `logic sum_valid [N];` block (after line ~46):

```systemverilog
    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;
```

Hunk 2 — dut instance, add after `.sum_valid  (sum_valid)` (comma-terminate the previous line):

```systemverilog
        .sum_valid  (sum_valid),
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
```

Hunk 3 — helper task + scalar, after the `run_and_check` task's `endtask`:

```systemverilog
    //----------------------------------------------------------------
    // Perf-counter read (registered port: data lands one posedge
    // after perf_addr) and check helper
    //----------------------------------------------------------------
    int c_val;

    task automatic read_ctr(input [3:0] idx, output int val);
        @(negedge clk) perf_addr = idx;
        @(negedge clk);
        val = perf_data;
    endtask

    task automatic check_ctr(input string name, input int got, input int exp);
        if (got !== exp) begin
            errors++;
            $error("[T4 perf] %s = %0d, expected %0d", name, got, exp);
        end else begin
            $display("PASS [T4 perf] %s = %0d", name, got);
        end
    endtask
```

Hunk 4 — init lines in the initial block (with the other zero-inits) and the T4 body. Init:

```systemverilog
        perf_clear = 0; perf_addr = '0;
```

T4 body, inserted after T3's late_results check and before the `//--- summary` block:

```systemverilog
        //--- T4: WS perf counters (map: 0 total, 1 load, 2 stream,
        //        3 flush, 5 idle, 7 tile_swaps; 4/6 unused read 0) ----
        // clear first: T1-T3 issued 4 gos; after clear + one 1-vec run
        // tile_swaps == 1 proves both the clear and the strobe count
        @(negedge clk) perf_clear = 1;
        @(negedge clk) perf_clear = 0;

        for (int j = 0; j < 4; j++) begin
            y_exp[j] = 0;
            for (int i = 0; i < 4; i++)
                y_exp[j] = y_exp[j] + w1[i][j]*a1[i];
        end
        run_and_check(2, 7, 1, "T4 perf run");

        read_ctr(4'd1, c_val);
        check_ctr("load (N+2)", c_val, N+2);
        read_ctr(4'd2, c_val);
        check_ctr("stream (n_vecs+1)", c_val, 2);
        read_ctr(4'd3, c_val);
        check_ctr("flush (2N+1)", c_val, 2*N+1);
        read_ctr(4'd7, c_val);
        check_ctr("tile_swaps", c_val, 1);
        read_ctr(4'd4, c_val);
        check_ctr("unused idx 4", c_val, 0);
        read_ctr(4'd6, c_val);
        check_ctr("unused idx 6", c_val, 0);
        read_ctr(4'd9, c_val);
        check_ctr("out-of-range idx 9", c_val, 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `./regress.sh` (or just the one TB compile: `iverilog -g2012 -o build/t.vvp rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv tb/tb_ws_controller.sv`)
Expected: COMPILE FAILED — `perf_clear`/`perf_addr`/`perf_data` are not ports of `ws_unit`.

- [ ] **Step 3: Implement the RTL**

3a. Create `rtl/ws_perf_counters.sv` (complete file):

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// ws_perf_counters: WS category mapping over the existing (unchanged)
// perf_counters hardware. PURE OBSERVER, same as the OS instance.
//
//   WS register map (CONTRACT - mirrored in docs/perf_map.md; the
//   AXI-Lite PERF_SEL/PERF_DATA pair reads it):
//     0: total_cycles   every cycle since perf_clear was last released
//     1: load_cycles    cycles in W_LOAD   (= N+2 per tile)
//     2: stream_cycles  cycles in W_STREAM (= n_vecs+1 per tile)
//     3: flush_cycles   cycles in W_FLUSH  (= 2N+1 per tile)
//     4: unused         reads 0
//     5: idle_cycles    cycles in W_IDLE
//     6: unused         reads 0
//     7: tile_swaps     count of w_capture strobes (1 per accepted go)
//   total=0 / idle=5 / tile-count=7 keep the OS map's indices so host
//   code shares offsets. Saturating counters, registered read port,
//   out-of-range addresses read 0 (all inherited from perf_counters).
//====================================================================
module ws_perf_counters #(
    parameter int NCTR = 8,
    parameter int CW   = 32
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          perf_clear,

    // Moore flags from ws_controller (never decoded enum values)
    input  logic          in_load,
    input  logic          in_stream,
    input  logic          in_flush,
    input  logic          in_idle,
    input  logic          tile_swap,     // 1-cycle strobe per tile load

    input  logic [3:0]    perf_addr,
    output logic [CW-1:0] perf_data      // valid 1 cycle after perf_addr
);

    logic const_zero;
    assign const_zero = 1'b0;

    perf_counters #(
        .NCTR(NCTR), .CW(CW)
    ) u_ctrs (
        .clk        (clk),
        .rst_n      (rst_n),
        .perf_clear (perf_clear),
        .in_fetch   (in_load),      // index 1: load
        .in_feed    (in_stream),    // index 2: stream
        .in_settle  (in_flush),     // index 3: flush
        .in_drain   (const_zero),   // index 4: unused
        .in_idle    (in_idle),      // index 5: idle
        .chunk_done (const_zero),   // index 6: unused
        .tile_done  (tile_swap),    // index 7: tile_swaps
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
    );

endmodule
```

3b. `rtl/ws_controller.sv` — two hunks.

Hunk 1, port list: after `output logic busy,` add:

```systemverilog
    output logic                          busy,

    // perf Moore flags (in_load is load_mode; tile strobe is w_capture)
    output logic                          in_stream,
    output logic                          in_flush,
    output logic                          in_idle,
```

Hunk 2, outputs section: after `assign busy = (state != W_IDLE);` add:

```systemverilog
    assign in_stream  = (state == W_STREAM);
    assign in_flush   = (state == W_FLUSH);
    assign in_idle    = (state == W_IDLE);
```

3c. `rtl/ws_unit.sv` — four hunks.

Hunk 1, port list: after `output logic busy,` add:

```systemverilog
    // perf counter access (ws_perf_counters map - see that module)
    input  logic                          perf_clear,
    input  logic [3:0]                    perf_addr,
    output logic [31:0]                   perf_data,
```

Hunk 2, nets: after the `logic load_mode, w_capture, valid_feed;` line add:

```systemverilog
    logic                         in_stream, in_flush, in_idle;
```

Hunk 3, `u_ctrl` instance: after `.busy (busy),` add:

```systemverilog
        .in_stream  (in_stream),
        .in_flush   (in_flush),
        .in_idle    (in_idle),
```

Hunk 4, new instance after the `u_ctrl` instance's closing `);`:

```systemverilog
    ws_perf_counters u_perf (
        .clk        (clk),
        .rst_n      (rst_n),
        .perf_clear (perf_clear),
        .in_load    (load_mode),
        .in_stream  (in_stream),
        .in_flush   (in_flush),
        .in_idle    (in_idle),
        .tile_swap  (w_capture),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
    );
```

3d. `rtl/ws_quant_unit.sv` — two hunks.

Hunk 1, port list: after `output logic busy,` add:

```systemverilog
    // perf counter access (pass-through to ws_unit)
    input  logic                          perf_clear,
    input  logic [3:0]                    perf_addr,
    output logic [31:0]                   perf_data,
```

Hunk 2, `u_ws` instance: after `.busy (busy),` add:

```systemverilog
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
```

3e. `tb/tb_ws_quant_unit.sv` and `tb/tb_ws_n16.sv` — identical treatment in each: declare + tie off + connect.

Declarations (next to the other DUT I/O):

```systemverilog
    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;
```

Init lines (in the initial block with the other zero-inits):

```systemverilog
        perf_clear = 0; perf_addr = '0;
```

dut instance, after `.busy (busy),`:

```systemverilog
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
```

3f. `regress.sh` — extend the three WS lists (new files added as part of creating them). Replace:

```bash
run_tb tb_ws_controller  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         tb/tb_ws_controller.sv
```

with:

```bash
run_tb tb_ws_controller  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         tb/tb_ws_controller.sv
```

and in the `tb_ws_quant_unit` and `tb_ws_n16` entries, insert the same
`rtl/perf_counters.sv rtl/ws_perf_counters.sv \` line before their
`rtl/requant.sv ...` line.

- [ ] **Step 4: Run regress to verify green**

Run: `./regress.sh`
Expected: all 13 TBs PASS, `=== REGRESSION GREEN ===`, exit 0. tb_ws_controller log shows the seven `PASS [T4 perf]` lines.

- [ ] **Step 5: Commit**

```bash
git add rtl/ws_perf_counters.sv rtl/ws_controller.sv rtl/ws_unit.sv rtl/ws_quant_unit.sv tb/tb_ws_controller.sv tb/tb_ws_quant_unit.sv tb/tb_ws_n16.sv regress.sh
git commit -m "ws_perf_counters: WS category map over existing counter hw (AXI spec step 1)

Map: 0 total, 1 load, 2 stream, 3 flush, 5 idle, 7 tile_swaps (indices
0/5/7 shared with the OS map). Flush added vs spec draft - W_FLUSH is
a real state. Additive: ws_controller Moore flags, ws_unit/ws_quant_unit
perf ports, T4 budget checks in tb_ws_controller."
```

---

### Task 2: axi_lite_ctrl skeleton — CTRL/STATUS, single shared clock (spec commit step 2)

**Files:**
- Create: `rtl/axi_lite_ctrl.sv`, `tb/tb_axi_lite_ctrl.sv`
- Modify: `regress.sh`

**Interfaces:**
- Produces (v1, stable through Task 4): AXI-Lite subordinate ports named `s_axi_*` exactly as listed in the module below; core-side ports `core_clk`, `core_rst_n`, `core_go` (1-cycle pulse), `core_busy`. Byte addresses: CTRL=0x00 (W, bit0 go, self-clearing, clears done), STATUS=0x04 (R, bit0 busy, bit1 done sticky). Writes commit only when BOTH AW and W have been accepted (any arrival order); one outstanding transaction per direction; `wstrb` ignored (full-word writes assumed, control plane); bresp/rresp always OKAY. v1 assumption: TB ties `core_clk` to `s_axi_aclk` (documented in header; CDC lands in Task 3).
- Consumes: nothing from the WS stack yet — the TB fakes the core.

- [ ] **Step 1: Write the failing test — create tb/tb_axi_lite_ctrl.sv**

Complete file, given in two halves (concatenate; this exact file evolves in Tasks 3/4). It fails now because `axi_lite_ctrl` does not exist.

Half 1 — module, signals, DUT, clock, fake core, BFM tasks:

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// tb_axi_lite_ctrl v1 (spec step 2): protocol + CTRL/STATUS checks
// against a FAKE core (busy modeled in the TB), single shared clock
// (core_clk tied to aclk until the CDC step).
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
        .core_clk      (aclk),        // v1: single shared clock
        .core_rst_n    (aresetn),
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
    always @(posedge aclk)
        if (aresetn && core_go === 1'b1) go_seen = go_seen + 1;

    initial core_busy = 1'b0;
    always @(posedge aclk) begin
        if (aresetn && core_go === 1'b1) begin
            repeat (2) @(negedge aclk);
            core_busy = 1'b1;
            repeat (40) @(negedge aclk);
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
            if (v[bit_idx] === 1'b1) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++;
                $error("poll %s: STATUS bit %0d never set", name, bit_idx);
                return;
            end
        end
    endtask
```

Half 2 — main sequence:

```systemverilog
    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [31:0] rd;

    initial begin
        aresetn = 0;
        awvalid = 0; awaddr = '0; wvalid = 0; wdata = '0; wstrb = 4'hF;
        bready = 0; arvalid = 0; araddr = '0; rready = 0;

        repeat (3) @(negedge aclk);
        aresetn = 1;
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

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
```

- [ ] **Step 2: Run to verify it fails**

Run: `iverilog -g2012 -o build/tb_axi_lite_ctrl.vvp tb/tb_axi_lite_ctrl.sv`
Expected: FAIL — `axi_lite_ctrl` unknown module.

- [ ] **Step 3: Create rtl/axi_lite_ctrl.sv (v1, complete file)**

```systemverilog
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
```

- [ ] **Step 4: Add the regress.sh entry (new files, same change)**

After the `tb_ws_n16` entry, add:

```bash
# --- AXI-Lite control wrapper (control plane; CDC from spec step 3 on) ---
run_tb tb_axi_lite_ctrl  rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv
```

- [ ] **Step 5: Run regress to verify green**

Run: `./regress.sh`
Expected: 14 TBs PASS (new: tb_axi_lite_ctrl with P1-P5 PASS lines), `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv regress.sh
git commit -m "axi_lite_ctrl v1: CTRL/STATUS skeleton, single-clock (AXI spec step 2)

AW/W accepted in either order, commit on both, one outstanding txn per
direction; done sticky on busy fall, cleared by go. TB: fake core,
protocol checks P1-P5 incl. staggered channels and held-RREADY reads."
```

---

### Task 3: CDC on the CTRL/STATUS pair — genuinely asynchronous clocks (spec commit step 3)

**Files:**
- Create: `rtl/sync_2ff.sv`, `rtl/pulse_sync.sv`
- Modify: `rtl/axi_lite_ctrl.sv`, `tb/tb_axi_lite_ctrl.sv`, `regress.sh`

**Interfaces:**
- Produces: `sync_2ff` (ports: `clk`, `rst_n`, `d`, `q` — 2-flop level synchronizer, single bit) and `pulse_sync` (ports: `src_clk`, `src_rst_n`, `src_pulse`, `dst_clk`, `dst_rst_n`, `dst_pulse` — toggle-and-synchronize; source pulses must be spaced more than ~3 destination cycles apart, which AXI transaction pacing guarantees here). Task 4's `bus_handshake` follows the same toggle idiom.
- `axi_lite_ctrl` port list is UNCHANGED — `core_clk` becomes a real second clock; the TB now drives it asynchronously.

- [ ] **Step 1: Make the test demand real CDC — rework tb/tb_axi_lite_ctrl.sv clocks**

Three hunks (test-first: this makes the v1 single-clock DUT fail its go/done checks under async clocks — expected to fail until Step 3):

Hunk 1 — replace the single-clock DUT hookup and clock gen. `core_clk`/`core_rst_n` connections become:

```systemverilog
        .core_clk      (cclk),
        .core_rst_n    (crst_n),
```

and after `always #5 aclk = ~aclk;` add:

```systemverilog
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
```

(Declare `cclk`/`crst_n` before the dut instance — Icarus needs declaration before use in the port list; put the `initial forever` block after the clock comment where shown.)

Hunk 2 — the fake core moves to the core clock. Replace both `always @(posedge aclk)` fake-core blocks' clock/edge references: `posedge aclk` -> `posedge cclk`, `negedge aclk` -> `negedge cclk` (BFM tasks stay on aclk). The header comment's v1 line "single shared clock" becomes "aclk 100 MHz vs jittered ~263 MHz cclk, asynchronous".

Hunk 3 — reset release covers both domains. In the initial block, replace `aresetn = 0;` with `aresetn = 0; crst_n = 0;` and `aresetn = 1;` with:

```systemverilog
        aresetn = 1;
        crst_n  = 1;
```

Also wrap the P2-P5 sequence in a phase-sweep loop (P1 stays outside, before the loop):

```systemverilog
        for (int it = 0; it < 3; it++) begin
            $display("--- async iteration %0d ---", it);
            ... P2 through P5 exactly as before ...
        end
```

(The `go_base = go_seen;` delta pattern already makes P2/P3/P5 loop-safe. Inside the loop, P4's done-still-set precondition holds each iteration: P2 leaves done=1, P3 writes have go bit 0.)

- [ ] **Step 2: Run to verify it fails (or is at least metastability-unsafe)**

Run: `./regress.sh`
Expected: tb_axi_lite_ctrl FAILs (go pulse missed/duplicated by the async fake core, or busy/done polling misses). NOTE: because simulation has no real metastability, a raw-wire crossing can occasionally pass by luck — if it passes, proceed anyway; the CDC rework is mandated by the spec, and Step 4 re-proves green. Do not skip Step 3 on a lucky pass.

- [ ] **Step 3: Create the CDC primitives and rework axi_lite_ctrl internals**

3a. `rtl/sync_2ff.sv` (complete file):

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// sync_2ff: 2-flop level synchronizer, single bit. For quasi-static
// levels only (busy) - never for pulses or multi-bit values.
//====================================================================
module sync_2ff (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);

    (* ASYNC_REG = "TRUE" *) logic meta;
    (* ASYNC_REG = "TRUE" *) logic sync1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            meta  <= 1'b0;
            sync1 <= 1'b0;
        end else begin
            meta  <= d;
            sync1 <= meta;
        end
    end

    assign q = sync1;

endmodule
```

3b. `rtl/pulse_sync.sv` (complete file):

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// pulse_sync: single-cycle pulse crossing via toggle-and-synchronize.
// Source toggles a flop per pulse; destination synchronizes the
// toggle and edge-detects it. Source pulses must be spaced more than
// ~3 destination cycles apart or toggles can be swallowed (AXI
// transaction pacing guarantees that for go/done here).
//====================================================================
module pulse_sync (
    input  logic src_clk,
    input  logic src_rst_n,
    input  logic src_pulse,
    input  logic dst_clk,
    input  logic dst_rst_n,
    output logic dst_pulse
);

    logic req_t;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n)
            req_t <= 1'b0;
        else if (src_pulse)
            req_t <= ~req_t;
    end

    (* ASYNC_REG = "TRUE" *) logic meta;
    (* ASYNC_REG = "TRUE" *) logic sync1;
    logic sync2;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            meta  <= 1'b0;
            sync1 <= 1'b0;
            sync2 <= 1'b0;
        end else begin
            meta  <= req_t;
            sync1 <= meta;
            sync2 <= sync1;
        end
    end

    assign dst_pulse = sync1 ^ sync2;

endmodule
```

3c. `rtl/axi_lite_ctrl.sv` v2 — header note becomes "TWO ASYNCHRONOUS DOMAINS: go via pulse_sync, busy via sync_2ff, done edge detected in the core domain and crossed via pulse_sync". Internal hunks:

Replace the `core_go <= go_write;` line (and drop `core_go` from that block's reset list) — `core_go` is no longer driven by the write-channel block. After the write-channel always block add:

```systemverilog
    //----------------------------------------------------------------
    // CDC: go pulse into the core domain
    //----------------------------------------------------------------
    logic go_pulse_axi;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn)
            go_pulse_axi <= 1'b0;
        else
            go_pulse_axi <= go_write;
    end

    pulse_sync u_cdc_go (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_pulse (go_pulse_axi),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_pulse (core_go)
    );

    //----------------------------------------------------------------
    // CDC: busy level into the AXI domain
    //----------------------------------------------------------------
    logic busy_axi;

    sync_2ff u_cdc_busy (
        .clk   (s_axi_aclk),
        .rst_n (s_axi_aresetn),
        .d     (core_busy),
        .q     (busy_axi)
    );

    //----------------------------------------------------------------
    // done: busy falling edge detected in the CORE domain (exact),
    // crossed as a pulse, sticky in the AXI domain until a go write
    //----------------------------------------------------------------
    logic core_busy_q;
    logic done_pulse_axi;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n)
            core_busy_q <= 1'b0;
        else
            core_busy_q <= core_busy;
    end

    logic core_done_pulse;
    assign core_done_pulse = core_busy_q && !core_busy;

    pulse_sync u_cdc_done (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_pulse (core_done_pulse),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_pulse (done_pulse_axi)
    );
```

The v1 done block loses its `busy_q` (delete it) and becomes:

```systemverilog
    logic done_ff;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn)
            done_ff <= 1'b0;
        else if (go_write)
            done_ff <= 1'b0;
        else if (done_pulse_axi)
            done_ff <= 1'b1;
    end
```

And the STATUS read line switches from raw `core_busy` to the synchronized level:

```systemverilog
                    A_STATUS: s_axi_rdata <= {30'b0, done_ff, busy_axi};
```

3d. `regress.sh` — extend the entry:

```bash
run_tb tb_axi_lite_ctrl  rtl/sync_2ff.sv rtl/pulse_sync.sv \
                         rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv
```

- [ ] **Step 4: Run regress to verify green**

Run: `./regress.sh`
Expected: 14 TBs PASS, tb_axi_lite_ctrl shows three `--- async iteration N ---` blocks each with P2-P5 PASS lines. `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv regress.sh
git commit -m "axi_lite_ctrl v2: real CDC on CTRL/STATUS (AXI spec step 3)

go via pulse_sync (toggle+sync+edge), busy via sync_2ff, done edge
detected in the core domain and crossed as a pulse. TB core clock is
now genuinely asynchronous (~263 MHz with per-half-cycle random
jitter) and the P2-P5 suite loops 3x to sweep phase alignments."
```

---

### Task 4: full register set + integration TB (spec commit step 4)

**Files:**
- Create: `rtl/bus_handshake.sv`
- Modify: `rtl/axi_lite_ctrl.sv` (v3 — full rewrite, complete file below), `tb/tb_axi_lite_ctrl.sv` (v3 — full rewrite, complete file in Step 5), `regress.sh`

**Interfaces:**
- Consumes: `ws_quant_unit` at N=4/K=4/DEPTH=64/VEC_W=4 (Task 1's version with perf ports); cycle budgets load=N+2, stream=n_vecs+1, flush=2N+1, tile_swaps=1/go; `sync_2ff` and `pulse_sync` from Task 3.
- Produces: `bus_handshake #(W)` (ports `src_clk, src_rst_n, src_data[W-1:0], src_send, src_busy, dst_clk, dst_rst_n, dst_data[W-1:0], dst_valid`) — toggle req/ack value crossing, source data captured atomically in the source domain and held until acked, so the destination NEVER sees a torn value. `axi_lite_ctrl` final core-side ports: `core_go`, `core_busy`, `core_n_vecs[VEC_W-1:0]`, `core_relu_en`, `core_scale_mult[SCALE_WIDTH-1:0]` (signed), `core_shift_amt[SHIFT_WIDTH-1:0]`, `core_perf_addr[3:0]`, `core_perf_data[31:0]`. REQUANT_CFG packing: `[15:0]=scale_mult, [20:16]=shift_amt, [21]=relu_en` (22 bits — fits one register, no split needed).

- [ ] **Step 1: Create rtl/bus_handshake.sv (complete file)**

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// bus_handshake: multi-bit value crossing via toggle req/ack.
//   NEVER synchronize the bits of a bus independently - different
//   bits resolve on different edges and the destination can see a
//   torn value. Here the source captures src_data into a holding
//   register and toggles req in the SAME source-domain edge; the
//   destination synchronizes req, and only on the synchronized edge
//   samples the holding register (guaranteed stable: the source
//   ignores src_send while src_busy, i.e. until the ack returns).
//   Round trip is ~3 dst + ~3 src cycles; callers queue re-sends
//   behind src_busy (see axi_lite_ctrl's dirty-flag pattern).
//====================================================================
module bus_handshake #(
    parameter int W = 32
)(
    input  logic         src_clk,
    input  logic         src_rst_n,
    input  logic [W-1:0] src_data,
    input  logic         src_send,     // 1-cycle pulse; ignored while src_busy
    output logic         src_busy,
    input  logic         dst_clk,
    input  logic         dst_rst_n,
    output logic [W-1:0] dst_data,     // registered; holds last delivered value
    output logic         dst_valid     // 1-cycle pulse per delivery
);

    //----------------------------------------------------------------
    // Source: capture + toggle request
    //----------------------------------------------------------------
    logic [W-1:0] hold_q;
    logic         req_t;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            hold_q <= '0;
            req_t  <= 1'b0;
        end else if (src_send && !src_busy) begin
            hold_q <= src_data;
            req_t  <= ~req_t;
        end
    end

    //----------------------------------------------------------------
    // Destination: sync request, capture on edge, toggle ack
    //----------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic req_meta;
    (* ASYNC_REG = "TRUE" *) logic req_s1;
    logic req_s2;
    logic ack_t;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            req_meta  <= 1'b0;
            req_s1    <= 1'b0;
            req_s2    <= 1'b0;
            ack_t     <= 1'b0;
            dst_data  <= '0;
            dst_valid <= 1'b0;
        end else begin
            req_meta  <= req_t;
            req_s1    <= req_meta;
            req_s2    <= req_s1;
            dst_valid <= 1'b0;
            if (req_s1 != req_s2) begin
                dst_data  <= hold_q;   // stable: source blocked until ack
                dst_valid <= 1'b1;
                ack_t     <= ~ack_t;
            end
        end
    end

    //----------------------------------------------------------------
    // Source: sync ack; busy while a request is outstanding
    //----------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic ack_meta;
    (* ASYNC_REG = "TRUE" *) logic ack_s1;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            ack_meta <= 1'b0;
            ack_s1   <= 1'b0;
        end else begin
            ack_meta <= ack_t;
            ack_s1   <= ack_meta;
        end
    end

    assign src_busy = (req_t != ack_s1);

endmodule
```

- [ ] **Step 2: Rewrite rtl/axi_lite_ctrl.sv (v3, complete file)**

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// axi_lite_ctrl v3 (AXI spec step 4): full single-tile register file
// across TWO ASYNCHRONOUS DOMAINS (s_axi_aclk / core_clk).
//
// Register map (byte offsets; word decode on addr[5:2]):
//   0x00 CTRL        W  [0] go - self-clearing pulse, clears done
//   0x04 STATUS      R  [0] busy (sync_2ff), [1] done (sticky)
//   0x08 N_VECS      W  [VEC_W-1:0]; AXI-side readback supported
//   0x0C REQUANT_CFG W  [15:0] scale_mult, [20:16] shift_amt,
//                       [21] relu_en (22 bits - one register)
//   0x10 PERF_SEL    W  [3:0] counter index; readback supported
//   0x14 PERF_DATA   R  selected counter, refreshed once per PERF_SEL
//                       write (bounded ~4 aclk + ~6 core clks later -
//                       hosts write PERF_SEL, wait, then read)
//
// CDC per signal class (spec section 2): go/done pulse_sync, busy
// sync_2ff, every multi-bit value bus_handshake (req/ack, source
// holds value until acked - never parallel per-bit 2-flop chains).
// Dirty-flag relaunch: a register rewritten while its handshake is
// in flight is re-sent when the ack returns, so the core converges
// on the latest value and never observes a torn one.
//
// Single-tile scope: w_tile_idx/a_base have no registers (spec map);
// the integration ties them to 0. First extension for WS tiling.
// Control plane only - zero DSPs expected in synth (a DSP here is a
// bug signal). Weight/activation data movement is NOT here (AXI-HP/
// DMA, separate work).
//====================================================================
module axi_lite_ctrl #(
    parameter int VEC_W       = 4,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
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
    output logic                          core_go,
    input  logic                          core_busy,
    output logic [VEC_W-1:0]              core_n_vecs,
    output logic                          core_relu_en,
    output logic signed [SCALE_WIDTH-1:0] core_scale_mult,
    output logic [SHIFT_WIDTH-1:0]        core_shift_amt,
    output logic [3:0]                    core_perf_addr,
    input  logic [31:0]                   core_perf_data
);

    localparam int CFG_W = SCALE_WIDTH + SHIFT_WIDTH + 1;   // 22

    // word address = byte address [5:2]
    localparam logic [3:0] A_CTRL   = 4'h0;
    localparam logic [3:0] A_STATUS = 4'h1;
    localparam logic [3:0] A_NVECS  = 4'h2;
    localparam logic [3:0] A_CFG    = 4'h3;
    localparam logic [3:0] A_PSEL   = 4'h4;
    localparam logic [3:0] A_PDATA  = 4'h5;

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

    // AXI-side register file + handshake launch state
    logic [VEC_W-1:0] nvecs_reg;
    logic [CFG_W-1:0] cfg_reg;
    logic [3:0]       psel_reg;
    logic nvecs_dirty, cfg_dirty, psel_dirty;
    logic nvecs_send,  cfg_send,  psel_send;
    logic nvecs_hs_busy, cfg_hs_busy, psel_hs_busy;
    logic go_pulse_axi;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            aw_got       <= 1'b0;
            w_got        <= 1'b0;
            s_axi_bvalid <= 1'b0;
            waddr_q      <= '0;
            wdata_q      <= '0;
            nvecs_reg    <= '0;
            cfg_reg      <= '0;
            psel_reg     <= '0;
            nvecs_dirty  <= 1'b0;
            cfg_dirty    <= 1'b0;
            psel_dirty   <= 1'b0;
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            go_pulse_axi <= 1'b0;
        end else begin
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            go_pulse_axi <= go_write;

            if (s_axi_awvalid && s_axi_awready) begin
                waddr_q <= s_axi_awaddr[5:2];
                aw_got  <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_q <= s_axi_wdata;
                w_got   <= 1'b1;
            end
            if (commit) begin
                case (waddr_q)
                    A_NVECS: begin
                        nvecs_reg   <= wdata_q[VEC_W-1:0];
                        nvecs_dirty <= 1'b1;
                    end
                    A_CFG: begin
                        cfg_reg   <= wdata_q[CFG_W-1:0];
                        cfg_dirty <= 1'b1;
                    end
                    A_PSEL: begin
                        psel_reg   <= wdata_q[3:0];
                        psel_dirty <= 1'b1;
                    end
                    default: ;      // CTRL handled via go_write; others ignored
                endcase
                s_axi_bvalid <= 1'b1;
                aw_got       <= 1'b0;
                w_got        <= 1'b0;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            // handshake launches: fire one cycle after the commit that
            // set dirty (holding registers are stable when sampled);
            // rewrites during flight relaunch after the ack returns
            if (nvecs_dirty && !nvecs_hs_busy && !nvecs_send) begin
                nvecs_send  <= 1'b1;
                nvecs_dirty <= 1'b0;
            end
            if (cfg_dirty && !cfg_hs_busy && !cfg_send) begin
                cfg_send  <= 1'b1;
                cfg_dirty <= 1'b0;
            end
            if (psel_dirty && !psel_hs_busy && !psel_send) begin
                psel_send  <= 1'b1;
                psel_dirty <= 1'b0;
            end
        end
    end

    //----------------------------------------------------------------
    // CDC: go pulse into the core domain
    //----------------------------------------------------------------
    pulse_sync u_cdc_go (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_pulse (go_pulse_axi),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_pulse (core_go)
    );

    //----------------------------------------------------------------
    // CDC: busy level into the AXI domain
    //----------------------------------------------------------------
    logic busy_axi;

    sync_2ff u_cdc_busy (
        .clk   (s_axi_aclk),
        .rst_n (s_axi_aresetn),
        .d     (core_busy),
        .q     (busy_axi)
    );

    //----------------------------------------------------------------
    // done: busy falling edge detected in the CORE domain (exact),
    // crossed as a pulse, sticky in the AXI domain until a go write
    //----------------------------------------------------------------
    logic core_busy_q;
    logic core_done_pulse;
    logic done_pulse_axi;
    logic done_ff;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n)
            core_busy_q <= 1'b0;
        else
            core_busy_q <= core_busy;
    end

    assign core_done_pulse = core_busy_q && !core_busy;

    pulse_sync u_cdc_done (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_pulse (core_done_pulse),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_pulse (done_pulse_axi)
    );

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn)
            done_ff <= 1'b0;
        else if (go_write)
            done_ff <= 1'b0;
        else if (done_pulse_axi)
            done_ff <= 1'b1;
    end

    //----------------------------------------------------------------
    // CDC: multi-bit values, AXI -> core (never per-bit sync chains)
    //----------------------------------------------------------------
    logic [VEC_W-1:0] nvecs_core;
    logic [CFG_W-1:0] cfg_core;
    logic nvecs_valid_core, cfg_valid_core, psel_valid_core;

    bus_handshake #(.W(VEC_W)) u_hs_nvecs (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (nvecs_reg),
        .src_send  (nvecs_send),
        .src_busy  (nvecs_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (nvecs_core),
        .dst_valid (nvecs_valid_core)
    );

    assign core_n_vecs = nvecs_core;

    bus_handshake #(.W(CFG_W)) u_hs_cfg (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (cfg_reg),
        .src_send  (cfg_send),
        .src_busy  (cfg_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (cfg_core),
        .dst_valid (cfg_valid_core)
    );

    assign core_scale_mult = cfg_core[SCALE_WIDTH-1:0];
    assign core_shift_amt  = cfg_core[SCALE_WIDTH+SHIFT_WIDTH-1:SCALE_WIDTH];
    assign core_relu_en    = cfg_core[CFG_W-1];

    bus_handshake #(.W(4)) u_hs_psel (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (psel_reg),
        .src_send  (psel_send),
        .src_busy  (psel_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_perf_addr),
        .dst_valid (psel_valid_core)
    );

    //----------------------------------------------------------------
    // Perf response: core_perf_data is valid one core cycle after
    // core_perf_addr lands (registered read port); wait two cycles
    // for margin, then hand the value back across. Same dirty-flag
    // relaunch so a hammered PERF_SEL still delivers a final answer.
    //----------------------------------------------------------------
    logic psel_v_d1, psel_v_d2;
    logic pdata_dirty, pdata_send;
    logic pdata_hs_busy;
    logic [31:0] pdata_shadow;
    logic pdata_valid_axi;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            psel_v_d1   <= 1'b0;
            psel_v_d2   <= 1'b0;
            pdata_dirty <= 1'b0;
            pdata_send  <= 1'b0;
        end else begin
            psel_v_d1  <= psel_valid_core;
            psel_v_d2  <= psel_v_d1;
            pdata_send <= 1'b0;
            if (psel_v_d2)
                pdata_dirty <= 1'b1;
            else if (pdata_dirty && !pdata_hs_busy && !pdata_send) begin
                pdata_send  <= 1'b1;
                pdata_dirty <= 1'b0;
            end
        end
    end

    bus_handshake #(.W(32)) u_hs_pdata (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_data  (core_perf_data),
        .src_send  (pdata_send),
        .src_busy  (pdata_hs_busy),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_data  (pdata_shadow),
        .dst_valid (pdata_valid_axi)
    );

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
                    A_STATUS: s_axi_rdata <= {30'b0, done_ff, busy_axi};
                    A_NVECS:  s_axi_rdata <= {{(32-VEC_W){1'b0}}, nvecs_reg};
                    A_CFG:    s_axi_rdata <= {{(32-CFG_W){1'b0}}, cfg_reg};
                    A_PSEL:   s_axi_rdata <= {28'b0, psel_reg};
                    A_PDATA:  s_axi_rdata <= pdata_shadow;
                    default:  s_axi_rdata <= '0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
```

- [ ] **Step 3: Run to confirm the gap**

Run: `./regress.sh`
Expected: tb_axi_lite_ctrl either COMPILE FAILS (bus_handshake not in its file list yet) or passes while exercising none of the new registers — either way the new behavior is untested. Proceed to the TB rewrite; it is the red/green cycle for this task.

- [ ] **Step 4: Rewrite tb/tb_axi_lite_ctrl.sv (v3, complete file)**

Given in two halves; concatenate.

Half 1 — module, clocks, DUT + real WS stack, monitors, golden helpers, BFM:

```systemverilog
`timescale 1ns / 1ps
//====================================================================
// tb_axi_lite_ctrl v3 (spec step 4): full integration - axi_lite_ctrl
// driving a REAL ws_quant_unit (N=4) across genuinely asynchronous
// clocks (aclk 100 MHz fixed; cclk ~263 MHz average with random
// per-half-cycle jitter, so the phase relationship sweeps through
// all alignments across the run - spec section 5).
//   P1: STATUS + unmapped addresses read 0 after reset
//   P2: staggered AW-first / W-first writes (ordering), no stray go
//   Functional suite, looped 3x for phase coverage:
//     F1: 15-vector identity-quant run. STATUS.busy observed mid-run,
//         done polled, exactly one core_go pulse, all 4x15 q8 results
//         exact vs golden, perf load/stream/flush/tile_swaps exact
//         via PERF_SEL/PERF_DATA (cumulative expectations).
//     F2: relu run - REQUANT_CFG provably reaches the requant lanes
//         (negative int32 results clamp to 0 at the int8 edge).
//     F3: back-to-back N_VECS writes - the core-side value is only
//         ever old/first/final (never torn), converges on the final
//         value, and a run then streams exactly that many vectors.
//   P3: held-RREADY read at the end - rvalid/rdata stable.
// Single-tile scope: w_tile_idx/a_base tied 0 (no registers in the
// spec map). Buffer preload uses the raw write ports on the core
// clock - data movement is the future DMA/AXI-HP path, not AXI-Lite.
//====================================================================
module tb_axi_lite_ctrl;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_PSEL   = 6'h10;
    localparam [5:0] ADDR_PDATA  = 6'h14;

    localparam int POLL_MAX = 50;
    // crossing-settle waits (generous vs the ~6 aclk + ~6 cclk actual)
    localparam int CFG_SETTLE_ACLKS  = 20;
    localparam int PERF_SETTLE_ACLKS = 20;

    //----------------------------------------------------------------
    // Clocks + resets (two genuinely asynchronous domains)
    //----------------------------------------------------------------
    logic aclk = 1'b0;
    logic cclk = 1'b0;
    logic aresetn, crst_n;

    always #5 aclk = ~aclk;    // 100 MHz AXI clock

    initial begin
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
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
    // Core-side nets (axi_lite_ctrl <-> ws_quant_unit) + WS extras
    //----------------------------------------------------------------
    logic                        core_go, core_busy;
    logic [VEC_W-1:0]            core_n_vecs;
    logic                        core_relu_en;
    logic signed [15:0]          core_scale_mult;
    logic [4:0]                  core_shift_amt;
    logic [3:0]                  core_perf_addr;
    logic [31:0]                 core_perf_data;

    logic [3:0]                  w_tile_idx;   // tied 0: single-tile scope
    logic [5:0]                  a_base;       // tied 0
    logic                        perf_clear;   // tied 0 (no AXI clear bit)
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [31:0]                 wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    int errors  = 0;
    int go_seen = 0;
    int go_base = 0;

    axi_lite_ctrl #(
        .VEC_W(VEC_W), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) dut (
        .s_axi_aclk      (aclk),
        .s_axi_aresetn   (aresetn),
        .s_axi_awvalid   (awvalid),
        .s_axi_awready   (awready),
        .s_axi_awaddr    (awaddr),
        .s_axi_wvalid    (wvalid),
        .s_axi_wready    (wready),
        .s_axi_wdata     (wdata),
        .s_axi_wstrb     (wstrb),
        .s_axi_bvalid    (bvalid),
        .s_axi_bready    (bready),
        .s_axi_bresp     (bresp),
        .s_axi_arvalid   (arvalid),
        .s_axi_arready   (arready),
        .s_axi_araddr    (araddr),
        .s_axi_rvalid    (rvalid),
        .s_axi_rready    (rready),
        .s_axi_rdata     (rdata),
        .s_axi_rresp     (rresp),
        .core_clk        (cclk),
        .core_rst_n      (crst_n),
        .core_go         (core_go),
        .core_busy       (core_busy),
        .core_n_vecs     (core_n_vecs),
        .core_relu_en    (core_relu_en),
        .core_scale_mult (core_scale_mult),
        .core_shift_amt  (core_shift_amt),
        .core_perf_addr  (core_perf_addr),
        .core_perf_data  (core_perf_data)
    );

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) u_ws (
        .clk        (cclk),
        .rst_n      (crst_n),
        .go         (core_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (core_n_vecs),
        .busy       (core_busy),
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

    //----------------------------------------------------------------
    // Core-domain monitors: go pulse counter, q8 collectors, and the
    // torn-value watch on core_n_vecs (spec section 5's back-to-back
    // write check: never an intermediate, neither-old-nor-new value)
    //----------------------------------------------------------------
    always @(posedge cclk)
        if (crst_n && core_go === 1'b1) go_seen = go_seen + 1;

    int got     [4][16];
    int got_cnt [4];

    always @(posedge cclk) begin
        if (crst_n) begin
            for (int j = 0; j < N; j++) begin
                if (q8_valid[j] === 1'b1) begin
                    got[j][got_cnt[j]] = q8_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                end
            end
        end
    end

    logic [3:0] nv_ok0, nv_ok1, nv_ok2;
    int nv_mon_en = 0;
    int nv_viol   = 0;

    always @(posedge cclk) begin
        if (nv_mon_en != 0) begin
            if (core_n_vecs !== nv_ok0 && core_n_vecs !== nv_ok1
                && core_n_vecs !== nv_ok2) begin
                nv_viol = nv_viol + 1;
                $error("[F3] torn N_VECS at core: %0d", core_n_vecs);
            end
        end
    end

    //----------------------------------------------------------------
    // Golden model (int source of truth; manual two's complement)
    //----------------------------------------------------------------
    int w  [4][4];     // resident weights w[i][j]
    int av [16][4];    // av[v][i]: activation vector v
    int y_exp;

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    function automatic int matvec(input int v, input int j);
        int y;
        y = 0;
        for (int i = 0; i < 4; i++)
            y = y + w[i][j]*av[v][i];
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

    // buffer preload (core clock - the future DMA path, not AXI-Lite)
    task automatic write_w(input int addr, input [31:0] data);
        @(negedge cclk) begin wr_en_w = 1; wr_addr_w = addr; wr_data_w = data; end
        @(negedge cclk) wr_en_w = 0;
    endtask

    task automatic write_a(input int addr, input [31:0] data);
        @(negedge cclk) begin wr_en_a = 1; wr_addr_a = addr; wr_data_a = data; end
        @(negedge cclk) wr_en_a = 0;
    endtask
```

Half 2 — BFM, perf/run helpers, main sequence:

```systemverilog
    //----------------------------------------------------------------
    // AXI-Lite BFM (drive negedge, sample posedge) - identical to the
    // Task 2/3 version
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
            if (v[bit_idx] === 1'b1) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++;
                $error("poll %s: STATUS bit %0d never set", name, bit_idx);
                return;
            end
        end
    endtask

    //----------------------------------------------------------------
    // Perf readback over AXI (PERF_SEL write, bounded wait, PERF_DATA
    // read). Expectations are CUMULATIVE since reset - budgets from
    // tb_ws_controller T4: load=N+2, stream=nv+1, flush=2N+1, 1 swap
    //----------------------------------------------------------------
    int exp_load = 0, exp_stream = 0, exp_flush = 0, exp_swaps = 0;

    task automatic perf_read(input [3:0] idx, output [31:0] val);
        axi_write(ADDR_PSEL, {28'b0, idx});
        repeat (PERF_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PDATA, val);
    endtask

    task automatic perf_check(input [3:0] idx, input int exp,
                              input string name);
        logic [31:0] v;
        perf_read(idx, v);
        if (v !== exp) begin
            errors++;
            $error("[%s] perf[%0d] = %0d, expected %0d", name, idx, v, exp);
        end else begin
            $display("PASS [%s] perf[%0d] = %0d", name, idx, v);
        end
    endtask

    //----------------------------------------------------------------
    // One full run through the register file: program N_VECS +
    // REQUANT_CFG (identity scale, optional relu), go, busy/done via
    // STATUS, values vs golden, perf counters vs cumulative budget
    //----------------------------------------------------------------
    task automatic run_via_axi(input int nv, input int relu,
                               input int check_busy, input string name);
        logic [31:0] st;
        logic [31:0] cfgword;

        for (int j = 0; j < N; j++) got_cnt[j] = 0;

        cfgword = '0;
        cfgword[0] = 1'b1;                 // scale_mult = 1, shift = 0
        if (relu != 0) cfgword[21] = 1'b1; // relu_en

        axi_write(ADDR_NVECS, nv);
        axi_write(ADDR_CFG, cfgword);
        axi_read(ADDR_NVECS, st);          // AXI-side readback sanity
        if (st !== nv) begin
            errors++; $error("[%s] N_VECS readback = %0d, wrote %0d", name, st, nv);
        end
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);   // let values cross

        go_base = go_seen;
        axi_write(ADDR_CTRL, 32'h1);

        if (check_busy != 0) begin
            // busy_axi is visible ~4 aclk after the go commit; a
            // 15-vec run holds busy ~31 cclk (~118 ns = ~12 aclk):
            // sampling ~7-9 aclk after the commit lands mid-window
            repeat (2) @(negedge aclk);
            axi_read(ADDR_STATUS, st);
            if (st[0] !== 1'b1) begin
                errors++; $error("[%s] STATUS.busy not observed mid-run", name);
            end else $display("PASS [%s] STATUS.busy observed", name);
        end

        poll_status_bit(1, name);
        repeat (10) @(negedge aclk);

        if (go_seen - go_base !== 1) begin
            errors++;
            $error("[%s] core_go pulses = %0d, expected 1", name, go_seen - go_base);
        end

        for (int j = 0; j < N; j++) begin
            if (got_cnt[j] !== nv) begin
                errors++;
                $error("[%s] column %0d: %0d results, expected %0d",
                       name, j, got_cnt[j], nv);
            end else begin
                for (int v = 0; v < nv; v++) begin
                    y_exp = clamp8(matvec(v, j), relu);
                    if (got[j][v] !== y_exp) begin
                        errors++;
                        $error("[%s] q8 y%0d[%0d] = %0d, expected %0d",
                               name, v, j, got[j][v], y_exp);
                    end
                end
            end
        end
        $display("PASS [%s] %0d vectors x %0d columns checked", name, nv, N);

        exp_load   = exp_load + (N + 2);
        exp_stream = exp_stream + (nv + 1);
        exp_flush  = exp_flush + (2*N + 1);
        exp_swaps  = exp_swaps + 1;
        perf_check(4'd1, exp_load,   name);
        perf_check(4'd2, exp_stream, name);
        perf_check(4'd3, exp_flush,  name);
        perf_check(4'd7, exp_swaps,  name);
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [31:0] rd;

    initial begin
        aresetn = 0; crst_n = 0;
        awvalid = 0; awaddr = '0; wvalid = 0; wdata = '0; wstrb = 4'hF;
        bready = 0; arvalid = 0; araddr = '0; rready = 0;
        w_tile_idx = '0; a_base = '0; perf_clear = 0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        nv_ok0 = '0; nv_ok1 = '0; nv_ok2 = '0;

        // weights clip both ways at identity quant; 16 deterministic
        // activation vectors spread over [-9, 9]
        w[0][0]=9;  w[0][1]=-8; w[0][2]=2;  w[0][3]=-9;
        w[1][0]=-7; w[1][1]=9;  w[1][2]=-1; w[1][3]=8;
        w[2][0]=1;  w[2][1]=-2; w[2][2]=3;  w[2][3]=-1;
        w[3][0]=8;  w[3][1]=-9; w[3][2]=1;  w[3][3]=9;
        for (int v = 0; v < 16; v++)
            for (int i = 0; i < 4; i++)
                av[v][i] = ((v*7 + i*3) % 19) - 9;

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        // preload weight tile 0 (words 0..3) + 16 vectors (words 0..15)
        for (int i = 0; i < 4; i++)
            write_w(i, pack4(w[i][0], w[i][1], w[i][2], w[i][3]));
        for (int v = 0; v < 16; v++)
            write_a(v, pack4(av[v][0], av[v][1], av[v][2], av[v][3]));

        //--- P1: reset state ------------------------------------------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS reads 0 after reset");
        axi_read(6'h18, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] unmapped 0x18 = %h, expected 0", rd);
        end else $display("PASS [P1] unmapped address reads 0");

        //--- P2: staggered channel ordering, no stray go ---------------
        go_base = go_seen;
        axi_write_aw_first(ADDR_CTRL, 32'h0);
        axi_write_w_first(ADDR_CTRL, 32'h0);
        repeat (10) @(negedge aclk);
        if (go_seen - go_base !== 0) begin
            errors++;
            $error("[P2] stray go pulse (delta=%0d)", go_seen - go_base);
        end else $display("PASS [P2] staggered AW/W writes, no stray go");

        //--- functional suite, 3 iterations for async phase coverage ---
        for (int it = 0; it < 3; it++) begin
            $display("--- async iteration %0d ---", it);

            run_via_axi(15, 0, 1, "F1 identity 15-vec");
            run_via_axi(1, 1, 0, "F2 relu 1-vec");

            //--- F3: back-to-back N_VECS writes, torn-value watch ------
            // second write lands while the first handshake is often
            // still in flight (round trip ~4-5 aclk vs ~5-6 aclk
            // between commits; jitter + 3 iterations cover both cases)
            nv_ok0 = 4'd1;      // current core value (F2's nv)
            nv_ok1 = 4'd2;      // first write
            nv_ok2 = 4'd5;      // final write
            nv_viol = 0;
            nv_mon_en = 1;
            axi_write(ADDR_NVECS, 32'd2);
            axi_write(ADDR_NVECS, 32'd5);
            repeat (20) @(negedge aclk);
            nv_mon_en = 0;
            errors = errors + nv_viol;
            if (core_n_vecs !== 4'd5) begin
                errors++;
                $error("[F3] core n_vecs = %0d, expected final 5", core_n_vecs);
            end
            if (nv_viol == 0 && core_n_vecs === 4'd5)
                $display("PASS [F3] b2b N_VECS: never torn, converged on 5");
            run_via_axi(5, 0, 0, "F3 stream n_vecs=5");
        end

        //--- P3: held-RREADY read (done still set from the last run) ---
        axi_read_slow(ADDR_STATUS, rd);
        if (rd[1] !== 1'b1) begin
            errors++; $error("[P3] STATUS.done lost during held read");
        end else $display("PASS [P3] rvalid/rdata held until rready");

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
```

- [ ] **Step 5: Update the regress.sh entry to the full stack**

```bash
run_tb tb_axi_lite_ctrl  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv
```

- [ ] **Step 6: Run regress to verify green**

Run: `./regress.sh`
Expected: 14 TBs PASS; tb_axi_lite_ctrl shows P1/P2 PASS, three `--- async iteration N ---` blocks each with F1 (incl. busy observation + 4 perf_check PASSes), F2, F3 PASS lines, then P3. `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add rtl/bus_handshake.sv rtl/axi_lite_ctrl.sv tb/tb_axi_lite_ctrl.sv regress.sh
git commit -m "axi_lite_ctrl v3: full register set + WS integration TB (AXI spec step 4)

N_VECS / REQUANT_CFG / PERF_SEL cross via bus_handshake (req/ack,
dirty-flag relaunch - core converges on latest value, never torn;
certified by the F3 back-to-back-write monitor). PERF_DATA is a
round-trip shadow (~10 aclk bound). TB drives a real ws_quant_unit
across jittered async clocks: golden q8 values, relu propagation,
exact perf budgets via the AXI path."
```

---

### Task 5: contract docs, report, final green, push

**Files:**
- Modify: `docs/perf_map.md` (append — Read it first; the existing OS map is a Phase 3 contract, do NOT renumber or edit it)
- Create: `reports/report-axi-lite-ctrl.md` (gitignored — reports/ is not committed)

**Interfaces:**
- Consumes: everything above, all green.
- Produces: the written contracts hosts/Python will code against, and the Vivado handoff list for the human.

- [ ] **Step 1: Append the WS section to docs/perf_map.md**

Read the file, then append (adjusting heading level to match the existing document):

```markdown
## WS counter map (ws_perf_counters instance inside ws_unit)

Same hardware (`perf_counters`), WS categories. Read over AXI-Lite via
PERF_SEL (0x10) / PERF_DATA (0x14) in `axi_lite_ctrl`.

| idx | name          | meaning                                    |
|-----|---------------|--------------------------------------------|
| 0   | total_cycles  | every cycle since perf_clear release        |
| 1   | load_cycles   | cycles in W_LOAD (= N+2 per tile)           |
| 2   | stream_cycles | cycles in W_STREAM (= n_vecs+1 per tile)    |
| 3   | flush_cycles  | cycles in W_FLUSH (= 2N+1 per tile)         |
| 4   | (unused)      | reads 0                                     |
| 5   | idle_cycles   | cycles in W_IDLE                            |
| 6   | (unused)      | reads 0                                     |
| 7   | tile_swaps    | count of weight-tile loads (1 per go)       |

Indices 0/5/7 deliberately match the OS map (total/idle/tile count) so
host code can share offsets. Counters saturate; out-of-range reads 0.
No AXI perf_clear bit yet (counters count from core reset; hosts use
deltas) — first candidate for a CTRL bit when needed.
```

- [ ] **Step 2: Write reports/report-axi-lite-ctrl.md**

Per the standing workflow (report per spec, gitignored). Contents — write it from the actual results, covering:

- What was built per spec step (1: ws_perf_counters + map table; 2: skeleton; 3: CDC; 4: full register set), with the commit hashes.
- Register map as implemented (byte offsets, bit packings: REQUANT_CFG `[15:0] scale, [20:16] shift, [21] relu` = 22 bits, single register — spec's split-register contingency not needed).
- Deviations from spec: flush counter added (W_FLUSH is a real state the spec's category list missed); w_tile_idx/a_base tied to 0 (absent from the spec's map, single-tile scope) — first extension point for WS tiling; CDC primitives as three standalone files.
- CDC design summary: per-signal-class treatment, dirty-flag relaunch semantics, host contracts (write PERF_SEL then wait ~10 aclk before reading PERF_DATA; done is the poll target, not busy; config writes settle within ~6 aclk + ~6 cclk before go).
- Verification evidence: which TB certifies what (tb_ws_controller T4 = budget/category proof; tb_axi_lite_ctrl = protocol, async CDC, torn-value, golden-value, perf-over-AXI proofs), final regress output snippet.
- **Vivado handoff asks (human):** (a) re-run synthesis with the new modules in the WS build; (b) check HOLD slack explicitly, not just WNS — pre-CDC margin was 0.107 ns at N=16 and synchronizer chains are classic short-path hold risks; (c) confirm `axi_lite_ctrl`/CDC modules infer ZERO DSPs (a DSP here is a bug signal); (d) constraints: the aclk/cclk pair needs async-clock-group (or set_max_delay) treatment in XDC — human-owned, ASYNC_REG attributes are already in the RTL; (e) xsim cross-check of tb_axi_lite_ctrl if convenient (Icarus/xsim divergence has bitten before).

- [ ] **Step 3: Final full regression**

Run: `./regress.sh`
Expected: 14 TBs PASS, `=== REGRESSION GREEN ===`, exit 0.

- [ ] **Step 4: Commit and push**

```bash
git add docs/perf_map.md
git commit -m "docs: WS perf counter map contract (AXI-Lite PERF_SEL/PERF_DATA)"
git push origin main
```

(reports/ is gitignored — the report stays local by design.)

---

## Execution notes

- Tasks are strictly ordered (each is a spec commit gate); do not start a task until the previous one's regression is green and committed.
- If any TB is flaky across runs (async jitter means runs are not bit-identical), treat it as a real CDC/testbench bug and fix it — do not loosen a check to pass. The one sanctioned tolerance is Task 3 Step 2, where a raw-wire crossing may pass by luck.
- If Icarus rejects a construct in this plan (e.g. `return` in tasks, `(* ASYNC_REG *)` attributes), the fallback is: replace `return` with a `done` flag + `if` guard; attributes can be deleted for simulation but should stay for the human's Vivado run — prefer keeping them (Icarus parses attribute instances).
- Timeouts: every poll loop is bounded by POLL_MAX; if a TB hangs anyway, check the done/busy crossing first (a swallowed pulse_sync toggle means pulses spaced too closely).
