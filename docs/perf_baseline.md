# Perf baseline: E3 (16x16x16, m=n=k=4 tiles, 16 output tiles, 64 chunks)

Counters read through the perf_counters port after E3 in tb_tiling_engine
(AUTO_CLEAR: numbers are for the E3 run alone). Utilization =
feed_cycles / total_cycles.

## Prediction (written BEFORE measuring)

Controller v4 (no prefetch overlap), per chunk: FETCH = K issue + 1 landing
= 5 cycles, FEED = 4 cycles. Per output tile (4 chunks): 4*(5+4) = 36, plus
SETTLE 8 + DRAIN 4 = 48 busy cycles, plus ~2 engine-overhead idle cycles
(the T_GO cycle and the cycle T_BUSY takes to observe !mm_busy) ≈ 50
cycles/tile.

- fetch_cycles ≈ 20/tile x 16 = 320
- feed_cycles  =  16/tile x 16 = 256
- settle       = 128, drain = 64
- total ≈ 16x50 ≈ 800 + start/readout tail
- **predicted utilization ≈ 256/800 ≈ 32%** (the spec's a-priori band was
  35-45%; the cycle accounting above lands just below it because SETTLE=8
  and DRAIN=4 per tile weigh more at k_tiles=4 than the band assumed)

## Baseline (measured, controller v4, commit of step 1)

```
[PERF E3 16x16x16 random] total=801 fetch=320 feed=256 settle=128
                          drain=64 idle=38 chunks=64 tiles=16
[PERF E3 16x16x16 random] utilization = feed/total = 31.9%
```

Reconciliation: fetch/feed/settle/drain/chunks/tiles match the prediction
exactly (5 fetch + 4 feed cycles per chunk, 64 chunks). total=801 = 16x48
busy + 33 engine/idle cycles (2 per tile boundary + start/done edges).
**Utilization 31.9%** vs predicted ~32%. Note the counters are sampled
sequentially (1 cycle apart) during the dump, and total/idle keep counting
during the readout itself, so idle (read 5 cycles after total) carries a
few extra post-run idle cycles; the per-state sums are consistent with
total once that skew is accounted for.

## After double-buffering: prediction (written BEFORE measuring)

Controller v5 prefetches chunk c+1 into the shadow bank while chunk c
feeds. The prefetch needs K issue cycles + 1 landing cycle = 5, but only
K = 4 FEED cycles overlap it (the first issue can start on FEED's first
cycle, since indices register at the FEED-entry edge). So exactly ONE
un-hidden remainder cycle per interior chunk; the cold first chunk of
each walk still pays the full 5.

- fetch_cycles/tile = 5 (cold) + 3x1 (remainders) = 8 -> 128 total
- feed unchanged 256; settle 128; drain 64
- busy/tile = 8+16+8+4 = 36; total ≈ 16x38 ≈ 610
- **predicted utilization ≈ 256/610 ≈ 42%**

## After double-buffering (measured, controller v5)

```
[PERF E3 16x16x16 random] total=609 fetch=128 feed=256 settle=128
                          drain=64 idle=38 chunks=64 tiles=16
[PERF E3 16x16x16 random] utilization = feed/total = 42.0%
```

E6 (same size, independent data) reproduces the numbers identically and
asserts fetch_cycles(128) < feed_cycles(256) as the structural proof the
overlap engaged.

## Delta and reconciliation

| metric        | v4 baseline | v5 double-buffered | delta          |
|---------------|-------------|--------------------|----------------|
| total_cycles  | 801         | 609                | -192 (-24.0%)  |
| fetch_cycles  | 320         | 128                | -192 (-60.0%)  |
| feed_cycles   | 256         | 256                | 0 (unchanged)  |
| settle/drain  | 128 / 64    | 128 / 64           | 0 (untouched)  |
| utilization   | 31.9%       | 42.0%              | +10.1 points   |

Reconciliation: measured = predicted, exactly. fetch went from 20/tile
(5 x 4 chunks) to 8/tile = 5 (cold first chunk, nothing to overlap with)
+ 3 x 1 (interior chunks). The remainder is exactly ONE cycle per interior
chunk - not 0 and not 2 - because a prefetch needs K+1 = 5 cycles (K issue
beats + 1 registered-read landing cycle) while only K = 4 FEED cycles can
hide it: indices register at the FEED-entry edge, the first read issues on
FEED's first cycle, the K-th read issues on FEED's last cycle, and its
landing is the one cycle FEED cannot cover. Remainder 0 would need the
chunk_req window one cycle earlier still (inside the previous remainder
cycle - indices are not committed yet); remainder 2 was the cost of the
initially-considered "register indices at FEED entry, first issue one
cycle later" variant, avoided by starting the issue counter at the swap
edge itself. The gap to 100% is now dominated by SETTLE (128) + DRAIN (64)
+ cold fetches (80) + engine idle (38): the next lever is the tile-level
go/collect overlap noted in tiling_engine as a future optimization.
