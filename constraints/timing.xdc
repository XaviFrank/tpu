# timing.xdc - shared across synth tops. XDC does not allow Tcl control
# flow, so every clock is declared with -quiet port lookups: on a top
# that lacks the port, the lookup returns nothing and Vivado skips that
# constraint with a Critical Warning (harmless - check report_clocks to
# confirm the intended clocks actually applied to the run's top).

# Core-domain clock, matmul/WS unit tops (port "clk")
create_clock -period 10.000 -name clk [get_ports -quiet clk]

# Core-domain clock, axi_lite_ctrl standalone/OOC top (port "core_clk").
# Period is the exploration target (~400 MHz measured at N=16 WS synth);
# retune per run alongside the "clk" line above.
create_clock -period 2.500 -name core_clk [get_ports -quiet core_clk]

# PS-side AXI-Lite clock (pl_clk; 100 MHz default, 6.667 for 150 MHz)
create_clock -period 10.000 -name s_axi_aclk [get_ports -quiet s_axi_aclk]

# The AXI and core domains are genuinely asynchronous: cut cross-domain
# setup/hold analysis. CDC safety is structural in the RTL (sync_2ff /
# pulse_sync / bus_handshake, ASYNC_REG attributes already in source).
# After any synth touching these: check HOLD slack explicitly, not just
# WNS - pre-CDC hold margin was 0.107 ns.
set_clock_groups -asynchronous \
    -group [get_clocks -quiet {clk core_clk}] \
    -group [get_clocks -quiet s_axi_aclk]
