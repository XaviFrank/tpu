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

## After double-buffering (measured, controller v5)

_to be filled in step 2_
