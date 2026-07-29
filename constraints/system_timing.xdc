# system_timing.xdc - constraints for the FULL-SYSTEM Block Design
# build only. timing.xdc is the standalone/per-module counterpart and
# stays exactly as it is: it is correct and still required for every
# standalone check (matmul/WS unit tops, axi_lite_ctrl, ws_tpu_top at
# N=4 and N=16, ws_perf_counters, the poke/peek smoke path). The two
# files are alternatives, never both active on the same run.
#
# WHY THIS FILE EXISTS
# timing.xdc declares core_clk and s_axi_aclk with create_clock. That
# is right when they are genuinely free top-level ports. It is wrong
# once ws_tpu_top is stitched into the Block Design: there those nets
# already carry a clock definition inherited from the PS-generated PL
# clocks, so a second create_clock is a redefinition of the same
# physical clock - exactly TIMING-4 (invalid primary clock
# redefinition) and TIMING-27 (invalid primary clock on hierarchical
# pin). The knock-on is TIMING-6/7 (no common primary clock / no
# common node between related clocks): the clock-group statement then
# refers to clock objects that do not share a source with the real
# ones, so cross-domain paths stop being cut and get timed instead.
# That is the likely bulk of the 411 failing endpoints - CDC paths
# being analysed as if they were synchronous.
#
# So: NO create_clock in this file, for any clock. Nothing here
# declares a clock; everything here only describes relationships
# between clocks the Block Design already defines.

#---------------------------------------------------------------------
# OPEN QUESTION - RESOLVE BEFORE TRUSTING A RUN WITH THIS FILE
#
# Does the asynchronous clock grouping below still need to exist here,
# and does it name the right clocks?
#
# It almost certainly IS still needed. The AXI and core domains remain
# genuinely asynchronous, and on Zynq MPSoC the PL clocks usually come
# from one PS PLL, so Vivado treats them as RELATED and will happily
# time paths between them unless told otherwise. CDC safety in this
# design is structural (sync_2ff / pulse_sync / bus_handshake, with
# ASYNC_REG already in the source), which assumes those paths are cut.
# Removing the grouping entirely would leave every CDC path timed.
#
# What is NOT certain is how to NAME the clocks here, because after
# integration they are the BD's clocks, not ours - typically clk_pl_0
# / clk_pl_1, or clk_out* from a clocking wizard. Two forms are given
# below. The active one is name-independent: it asks for whatever
# clock actually reaches ws_tpu_top's clock pins. The commented one is
# the explicit-name form, which is easier to read but must be edited
# to match the real design.
#
# THE FAILURE MODE TO WATCH FOR: an empty -group is a SILENT no-op.
# If the object queries below match nothing, this file applies no
# grouping at all and says nothing about it - the TIMING-4/27
# redefinition warnings will be gone (because no clock is redeclared),
# while the cross-domain paths stay timed and the endpoint failures
# stay put. Do not read "TIMING-4 cleared" as "constraint working".
# Verify positively, with the checklist at the bottom of this file.
#---------------------------------------------------------------------

# Name-independent form: resolve the clocks from the pins of the
# ws_tpu_top instance itself, whatever the BD named it. Anchored on
# the cell's REF_NAME so an instance rename does not break it, and so
# a bare */s_axi_aclk filter cannot pick up every other AXI IP's clock
# pin in the system.
set_clock_groups -asynchronous \
    -group [get_clocks -quiet -of_objects \
              [get_pins -quiet -of_objects \
                 [get_cells -quiet -hier -filter {REF_NAME =~ ws_tpu_top*}] \
               -filter {NAME =~ */core_clk}]] \
    -group [get_clocks -quiet -of_objects \
              [get_pins -quiet -of_objects \
                 [get_cells -quiet -hier -filter {REF_NAME =~ ws_tpu_top*}] \
               -filter {NAME =~ */s_axi_aclk}]]

# Explicit-name alternative. Prefer this once report_clocks has told
# you the real names - it is far easier to review, and it fails loudly
# rather than silently when a name is wrong. Replace the names to
# match the design, then delete the block above.
#
# set_clock_groups -asynchronous \
#     -group [get_clocks clk_pl_0] \
#     -group [get_clocks clk_pl_1]

#---------------------------------------------------------------------
# CLOCK PERIOD - NOT SET HERE, AND WORTH CHECKING
#
# timing.xdc pins the core domain at 4.700 ns (212.8 MHz). This file
# cannot: the period now comes from the PL clock frequency configured
# in the PS block, and a create_clock here would be the very
# redefinition this file exists to avoid.
#
# Confirm the PS PL clock feeding core_clk is actually configured for
# ~212.8 MHz. If it is left at a default 100 MHz, timing will "pass"
# comfortably while the accelerator silently runs at under half the
# intended rate - a green report that means the opposite of what it
# looks like. Cross-check with report_clocks, not with the BD dialog.
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# VERIFICATION CHECKLIST for a run using this file
#
# 1. report_clocks
#    Exactly one definition per physical clock. No core_clk or
#    s_axi_aclk of ours alongside the PS clocks.
#
#    OBSERVED SIGNATURE WHEN THE FOLLOW-UP BELOW IS STILL UNDONE
#    (routed 2026-07-29, ws_tpu_system_wrapper): the Clock Summary
#    lists FOUR clocks where there should be two -
#      clk_pl_0    10.000 ns   (real, PS)
#      clk_pl_1     5.000 ns   (real, PS)
#      core_clk     4.700 ns   (OURS - duplicate of clk_pl_1)
#      s_axi_aclk  10.000 ns   (OURS - duplicate of clk_pl_0)
#    Vivado names it directly in TIMING-4 ("core_clk is defined
#    downstream of clock clk_pl_1") and TIMING-27 (created on pin
#    ws_tpu_top_0/inst/core_clk). Each physical clock carries two
#    unrelated definitions, so Vivado times each net against itself
#    and cannot relate the edges.
#
#    WHAT THAT LOOKS LIKE DOWNSTREAM, so it is not misdiagnosed:
#    every failing endpoint is INTER-clock between a duplicate pair,
#    while intra-clock timing is completely clean (core_clk +0.362 ns
#    over 8340 endpoints, 0 failing). The failures concentrate on
#    reset paths from proc_sys_reset_0 - 407 Recovery checks on /CLR
#    pins plus 4 Setup checks on /CE pins - which reads exactly like
#    an asynchronous reset crossing needing a false path. It is NOT.
#    proc_sys_reset_0 is clocked by clk_pl_1, which IS core_clk's
#    physical clock, so those are ordinary same-domain synchronous
#    reset paths. A set_false_path there would mask a legitimate
#    recovery/removal check and leave the duplication in place.
#    Fix the redefinition; do not constrain around it.
#
# 2. report_clock_interaction
#    The core and AXI domains must show as "Asynchronous Groups", not
#    "Timed". If they still read Timed, the grouping above matched
#    nothing - go to the explicit-name form.
#
# 3. Methodology report
#    TIMING-4, TIMING-6, TIMING-7 and TIMING-27 all clear.
#
# 4. Timing summary
#    WNS/TNS back to a healthy range (prior good full-context runs sat
#    around +0.2 to +0.5 ns), not the -1.126 ns / -394.964 ns being
#    seen now. CHECK HOLD TOO, not just setup - WHS has been the
#    shrinking number across integrations (0.107 -> 0.058 -> 0.055 ns)
#    and clock-group changes move hold as well as setup.
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# FOLLOW-UP THIS FILE DOES NOT AND CANNOT DO (Vivado project config)
#
# Creating this file resolves nothing on its own. The redefinition
# comes from timing.xdc still being active during ws_tpu_top's per-IP
# out-of-context synthesis pass, and constraint-set membership is a
# project setting, not something any .xdc can express. Someone has to,
# in Vivado:
#   - detach timing.xdc from whatever drives ws_tpu_top's per-IP
#     synthesis in the BD flow, so it stops declaring clocks there;
#   - make system_timing.xdc the active constraint set for impl_1 /
#     the full-system build;
#   - leave timing.xdc active for the standalone module runs, which
#     still need it unchanged.
# Until that is done, this file changes nothing about any run.
#---------------------------------------------------------------------
