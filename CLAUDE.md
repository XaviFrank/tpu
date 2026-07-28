# TPU Project — working rules
- Verification: run ./regress.sh (bash) — it self-activates the Icarus PATH. Do not invoke iverilog/vvp directly; the script's per-TB file lists are the source of truth.
- Known Icarus quirks in this environment: no whole-unpacked-array assignment, no expressions in port connections, unpacked-array element signedness is unreliable (golden models use int locals with manual two's complement). Write around these preemptively.
- Debug probes: $display blocks at module level; read output from build/<tb>.log, not the console (regress.sh filters non-error lines).
- regress.sh file lists are per-TB and explicit; new files must be added to it as part of creating them.
- RTL style: dull portable constructs only (no whole-array assigns, no inline sized casts, no expressions in port connections; wires shared across generate blocks are declared at module level).
- Reset style: control registers get async reset, datapath registers get none.
- TB conventions: drive negedge / sample posedge, !== comparisons, clear pulse after reset, self-checking with "ALL TESTS PASSED" banner.
- Never edit files under vivado/. Synthesis is run by the human in Vivado.
- Commit at every green regression with a descriptive message.