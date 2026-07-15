# TPU Project — working rules
- confirm that we should run ./regress.sh after every RTL or TB change. Green banner + exit 0 required before any task is "done", but human confirmation is vital. Pending human change.
- regress.sh file lists are per-TB and explicit; new files must be added to it as part of creating them.
- RTL style: dull portable constructs only (no whole-array assigns, no inline sized casts, no expressions in port connections; wires shared across generate blocks are declared at module level).
- Reset style: control registers get async reset, datapath registers get none.
- TB conventions: drive negedge / sample posedge, !== comparisons, clear pulse after reset, self-checking with "ALL TESTS PASSED" banner.
- Never edit files under vivado/. Synthesis is run by the human in Vivado.
- Commit at every green regression with a descriptive message.