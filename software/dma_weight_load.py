#!/usr/bin/env python3
"""dma_weight_load.py - autonomous weight delivery over AXI DMA.

Replaces poke's ROLE for real weight data on the single-tile
``ws_tpu_top`` build: the host writes a DRAM buffer, the DMA moves it,
and nothing walks the tile a lane at a time over AXI-Lite.

    PS DRAM  --HP0-->  axi_dma_0 MM2S  --AXI4-Stream 128b-->
    axi_stream_to_buf_ad_0  -->  ws_tpu_top_0.wr_*_w  -->  weight tile

WHAT MAKES THIS A REAL TEST AND NOT A WIRING DEMO
    The bytes are ``qproj_l0_tile0.json`` - one 16x16 tile of Qwen3's
    real layer-0 q_proj, already proven bit-exact on silicon through
    poke/peek. So a mismatch here is the DMA path's fault and nothing
    else's; there is no new golden to be wrong about. Synthesising a
    fresh test pattern would have thrown that away.

    Verification is ``peek``, which shares the controller's own buffer
    read port - it is the real tile_buffer, not a shadow copy.

HOST CONTRACT - both halves matter
    1. Never trigger a run while a transfer is in flight. The DMA
       writes the weight tile underneath the array otherwise, and the
       core reads a half-written tile: a confidently wrong answer, not
       an error. This script waits for MM2S idle before any ``go``.
    2. Never start a transfer while a run is in flight. THIS half is
       now enforced in hardware - ``buf_ready = !core_busy`` - but only
       on a bitstream built after that change. On the bitstream as of
       this writing ``buf_ready`` is tied to an xlconstant(1), so this
       script also checks ``STATUS[0]`` itself and refuses to start.
       Simulation says 14 of 16 beats land underneath a live run on the
       constant-1 wiring, so this is a measured hazard.

    DMA and poke are ALTERNATIVES, never concurrent: ws_tpu_top's poke
    mux gives the external wr_* ports priority, so while the adapter
    holds wr_en high, pokes to that buffer are silently lost.

ALIGNMENT
    The DMA is built with C_INCLUDE_MM2S_DRE=0, so the source address
    must be a multiple of the 128-bit stream width (16 bytes).
    ``pynq.allocate`` returns page-aligned physical memory, so this
    holds - but it is asserted rather than assumed, because a DRE-less
    unaligned transfer fails as an internal error rather than as
    anything legible.

USAGE (in the PYNQ notebook)
    from dma_weight_load import DmaWeightLoader
    ldr = DmaWeightLoader()                  # loads/attaches the overlay
    ldr.report(ldr.run())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Hardware constants. Sourced from ws_tpu_system.hwh, not from memory.
# ----------------------------------------------------------------------
TPU_BASE = 0xB000_0000          # ws_tpu_top_0    (4 KB REGISTER window)
DMA_BASE = 0xB001_0000          # axi_dma_0       (64 KB S_AXI_LITE)

# ws_tpu_top register map - reports/report-pynq-register-map.md
CTRL, STATUS, N_VECS, REQUANT_CFG = 0x00, 0x04, 0x08, 0x0C
PERF_SEL, PERF_DATA = 0x10, 0x14
POKE_ADDR, POKE_DATA = 0x20, 0x24
PEEK_ADDR, PEEK_DATA = 0x28, 0x2C

ST_BUSY, ST_DONE, ST_CFG_ERR, ST_POKE_BUSY, ST_POKE_ERR = 1, 2, 4, 8, 16

# PEEK address fields: {buf_sel[1:0], idx1[3:0], idx0[5:0]}
#   buf_sel 0 = weight buffer, 1 = activation, 2 = INT8 capture,
#           3 = INT32 capture.  For a buffer read idx1 is the 32-bit
#           lane and idx0 is the word address.
PEEK_BUF_W, PEEK_BUF_A, PEEK_CAP_I32 = 0, 1, 3

# AXI DMA v7.1, simple mode (C_INCLUDE_SG=0), MM2S channel only
MM2S_DMACR, MM2S_DMASR = 0x00, 0x04
MM2S_SA, MM2S_SA_MSB, MM2S_LENGTH = 0x18, 0x1C, 0x28
DMACR_RS, DMACR_RESET = 0x1, 0x4
DMASR_HALTED, DMASR_IDLE = 0x1, 0x2
DMASR_DMAINTERR, DMASR_DMASLVERR, DMASR_DMADECERR = 0x10, 0x20, 0x40
DMASR_ERR_MASK = DMASR_DMAINTERR | DMASR_DMASLVERR | DMASR_DMADECERR

# Geometry of this build
N = 16                          # array width / bytes per buffer word
TILE_WORDS = 16                 # a weight tile is 16 words
WORD_BYTES = N                  # 128 bits
LANES = 4                       # 32-bit lanes per buffer word
STREAM_ALIGN = WORD_BYTES       # no DRE -> source must be 16B aligned

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "out", "qproj_l0_tile0.json")


# ----------------------------------------------------------------------
@dataclass
class Result:
    """Everything the caller needs to judge the transfer, and nothing
    interpreted. ``report()`` does the judging, separately, so a raw
    result can be inspected when the verdict is surprising."""
    ok: bool = False
    stage: str = ""
    payload_bytes: int = 0
    phys_addr: int = 0
    dma_status: int = 0
    words_checked: int = 0
    words_bad: int = 0
    bad_detail: list = field(default_factory=list)
    ran_matmul: bool = False
    results_bad: int = 0
    result_detail: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class DmaWeightLoader:
    def __init__(self, bitfile=None, json_path=DEFAULT_JSON, overlay=None):
        self.json_path = json_path
        with open(json_path) as fh:
            self.spec = json.load(fh)

        if overlay is not None:
            self.ol = overlay
        else:
            from pynq import Overlay
            # download=False attaches to an already-configured PL rather
            # than re-downloading, which would reset the core mid-session.
            self.ol = Overlay(bitfile, download=(bitfile is not None)) \
                if bitfile else Overlay("ws_tpu_system.bit", download=False)

        self.tpu = self._find_ip(TPU_BASE, "ws_tpu_top")
        self.dma_ip = self._find_ip(DMA_BASE, "axi_dma")
        self.dma = getattr(self.ol, "axi_dma_0", None)

    # -- plumbing ------------------------------------------------------
    def _find_ip(self, base, hint):
        """Locate an IP by its physical base, falling back to a name
        hint. Matching on the address is what makes this survive an
        instance rename in the block design."""
        for name, desc in self.ol.ip_dict.items():
            if desc.get("phys_addr") == base:
                return getattr(self.ol, name)
        for name in self.ol.ip_dict:
            if hint in name:
                return getattr(self.ol, name)
        raise RuntimeError(
            f"no IP at {base:#010x} (hint {hint!r}); "
            f"available: {list(self.ol.ip_dict)}")

    def rd(self, off):
        return self.tpu.read(off)

    def wr(self, off, val):
        self.tpu.write(off, val)

    def dma_rd(self, off):
        return self.dma_ip.read(off)

    def dma_wr(self, off, val):
        self.dma_ip.write(off, val)

    # -- payload -------------------------------------------------------
    def tile_bytes(self):
        """The 256-byte DRAM image of the weight tile.

        The layout rule is 'buffer word i = row i; byte j = column j,
        byte 0 in bits[7:0]'. The DMA packs DRAM bytes into a beat
        little-endian - byte 0 of each 16-byte group lands in
        tdata[7:0] - so a flat little-endian byte image of the words IS
        the required beat layout, with no reordering anywhere.

        The tile is ALREADY transposed in the JSON (layout.transpose:
        word i, byte j = W_pt[j][i]); do not transpose again here.
        """
        words = sorted(self.spec["weight_words"], key=lambda w: w["addr"])
        if [w["addr"] for w in words] != list(range(TILE_WORDS)):
            raise ValueError(f"weight_words addrs are not 0..{TILE_WORDS-1}")
        buf = bytearray()
        for w in words:
            val = int(w["hex"], 16)
            buf += val.to_bytes(WORD_BYTES, "little")
        assert len(buf) == TILE_WORDS * WORD_BYTES
        return bytes(buf)

    def expected_words(self):
        """The same tile as 16 ints, for comparison against peek."""
        words = sorted(self.spec["weight_words"], key=lambda w: w["addr"])
        return [int(w["hex"], 16) for w in words]

    # -- peek ----------------------------------------------------------
    def peek(self, buf_sel, idx1, idx0):
        self.wr(PEEK_ADDR, (buf_sel << 10) | (idx1 << 6) | idx0)
        return self.rd(PEEK_DATA)          # refresh-on-address-write

    def peek_word(self, buf_sel, addr):
        """One whole 128-bit buffer word, reassembled from its lanes."""
        word = 0
        for lane in range(LANES):
            word |= self.peek(buf_sel, lane, addr) << (lane * 32)
        return word

    def peek_result(self, vec, col):
        raw = self.peek(PEEK_CAP_I32, col, vec)
        return raw - (1 << 32) if raw & 0x8000_0000 else raw

    # -- poke (activations only; weights come over DMA now) ------------
    def poke_word(self, buf_sel, addr, data128, poll_limit=10000):
        for lane in range(LANES):
            self.wr(POKE_ADDR, (buf_sel << 8) | (lane << 6) | addr)
            if lane == LANES - 1:
                n = 0
                while self.rd(STATUS) & ST_POKE_BUSY:
                    n += 1
                    if n > poll_limit:
                        raise RuntimeError("poke_busy never cleared")
            self.wr(POKE_DATA, (data128 >> (lane * 32)) & 0xFFFF_FFFF)

    # -- the transfer --------------------------------------------------
    def dma_reset(self):
        self.dma_wr(MM2S_DMACR, DMACR_RESET)
        for _ in range(10000):
            if not (self.dma_rd(MM2S_DMACR) & DMACR_RESET):
                return
        raise RuntimeError("MM2S soft reset never cleared")

    def dma_send_raw(self, phys_addr, nbytes, poll_limit=1000000):
        """Simple-mode MM2S. Deliberately raw rather than
        ``sendchannel.transfer``: the register sequence is what the
        eventual bare-metal / autonomous controller has to emit, and
        seeing it here means the notebook is not the only place it
        exists. Writing LENGTH is what starts the transfer, so it goes
        last."""
        if phys_addr % STREAM_ALIGN:
            raise ValueError(
                f"source {phys_addr:#x} not {STREAM_ALIGN}-byte aligned "
                "and the DMA has no DRE")
        self.dma_wr(MM2S_DMACR, DMACR_RS)                 # run
        for _ in range(10000):
            if not (self.dma_rd(MM2S_DMASR) & DMASR_HALTED):
                break
        else:
            raise RuntimeError("MM2S stayed halted after RS")
        self.dma_wr(MM2S_SA, phys_addr & 0xFFFF_FFFF)
        self.dma_wr(MM2S_SA_MSB, (phys_addr >> 32) & 0xFFFF_FFFF)
        self.dma_wr(MM2S_LENGTH, nbytes)                  # starts it

        n = 0
        while True:
            sr = self.dma_rd(MM2S_DMASR)
            if sr & DMASR_ERR_MASK:
                raise RuntimeError(f"MM2S error, DMASR={sr:#010x}")
            if sr & DMASR_IDLE:
                return sr
            n += 1
            if n > poll_limit:
                raise RuntimeError(f"MM2S never went idle, DMASR={sr:#010x}")

    # -- the whole thing -----------------------------------------------
    def run(self, verify_matmul=True):
        from pynq import allocate

        r = Result()
        try:
            r.stage = "precheck"
            st = self.rd(STATUS)
            if st & ST_BUSY:
                r.notes.append(
                    f"REFUSING: core busy (STATUS={st:#x}). A transfer "
                    "started now would land underneath a live run.")
                return r
            if st & ST_POKE_BUSY:
                r.notes.append(f"REFUSING: poke in flight (STATUS={st:#x})")
                return r

            r.stage = "allocate"
            payload = self.tile_bytes()
            r.payload_bytes = len(payload)
            buf = allocate(shape=(len(payload),), dtype="u1")
            try:
                buf[:] = bytearray(payload)
                buf.flush()          # HP0 is not coherent - push it out
                r.phys_addr = int(buf.physical_address)

                r.stage = "dma"
                self.dma_reset()
                r.dma_status = self.dma_send_raw(r.phys_addr, len(payload))
            finally:
                buf.freebuffer()

            r.stage = "verify-bits"
            exp = self.expected_words()
            for addr, e in enumerate(exp):
                got = self.peek_word(PEEK_BUF_W, addr)
                r.words_checked += 1
                if got != e:
                    r.words_bad += 1
                    if len(r.bad_detail) < 4:
                        r.bad_detail.append(
                            f"word {addr}: got {got:#034x} exp {e:#034x}")

            if verify_matmul and r.words_bad == 0:
                r.stage = "verify-matmul"
                r.ran_matmul = True
                avecs = sorted(self.spec["activation_words"],
                               key=lambda w: w["addr"])
                for w in avecs:
                    self.poke_word(PEEK_BUF_A, w["addr"], int(w["hex"], 16))
                while self.rd(STATUS) & ST_POKE_BUSY:
                    pass
                self.wr(N_VECS, self.spec["n_vecs"])
                self.wr(REQUANT_CFG, 0x1)          # mult=1, shift=0, relu off
                if self.rd(STATUS) & ST_POKE_ERR:
                    r.notes.append("poke_err set while loading activations")
                self.wr(CTRL, 0x1)                 # go (also clears done)
                n = 0
                while not (self.rd(STATUS) & ST_DONE):
                    n += 1
                    if n > 1_000_000:
                        raise RuntimeError("done never asserted")
                golden = self.spec["expected"]["golden_int32"]
                for j, g in enumerate(golden):
                    got = self.peek_result(0, j)
                    if got != g:
                        r.results_bad += 1
                        if len(r.result_detail) < 4:
                            r.result_detail.append(
                                f"y[0][{j}]: got {got} exp {g}")

            r.stage = "done"
            r.ok = (r.words_bad == 0 and r.results_bad == 0)
        except Exception as exc:                    # noqa: BLE001
            r.notes.append(f"{type(exc).__name__} during {r.stage}: {exc}")
        return r

    # -- output --------------------------------------------------------
    @staticmethod
    def report(r):
        print("=== DMA weight delivery ===")
        print(f"  payload      : {r.payload_bytes} B "
              f"({r.payload_bytes // WORD_BYTES} beats of {WORD_BYTES} B)")
        print(f"  phys addr    : {r.phys_addr:#012x}")
        print(f"  MM2S DMASR   : {r.dma_status:#010x}")
        print(f"  weight words : {r.words_checked - r.words_bad}"
              f"/{r.words_checked} exact")
        for d in r.bad_detail:
            print(f"      {d}")
        if r.ran_matmul:
            n = len(r.result_detail)
            print(f"  matmul       : {16 - r.results_bad}/16 outputs match "
                  "the JSON golden")
            for d in r.result_detail:
                print(f"      {d}")
        for note in r.notes:
            print(f"  NOTE: {note}")
        print("  RESULT       : " + ("PASS" if r.ok else "FAIL"))
        return r.ok


def selfcheck(json_path=DEFAULT_JSON):
    """Everything that can be verified WITHOUT the board. Run it before
    taking the notebook out, so a payload-construction bug never gets
    diagnosed as a DMA problem.

    Needs no pynq, no overlay, no hardware.
    """
    ldr = object.__new__(DmaWeightLoader)
    ldr.json_path = json_path
    with open(json_path) as fh:
        ldr.spec = json.load(fh)

    ok = True
    payload = ldr.tile_bytes()
    print("=== dma_weight_load selfcheck ===")

    n_ok = len(payload) == TILE_WORDS * WORD_BYTES
    ok &= n_ok
    print(f"  payload size          : {len(payload)} B "
          f"({'ok' if n_ok else 'EXPECTED %d' % (TILE_WORDS*WORD_BYTES)})")

    # The strongest available check: the byte image must reproduce the
    # poke_lanes, which are the exact 32-bit values already proven
    # bit-exact on silicon. If these agree, DMA delivers byte-identical
    # data to what poke delivered - so a hardware mismatch afterwards is
    # the transport's fault and cannot be the payload's.
    bad = 0
    for w in sorted(ldr.spec["weight_words"], key=lambda x: x["addr"]):
        grp = payload[w["addr"]*WORD_BYTES:(w["addr"]+1)*WORD_BYTES]
        for lane in range(LANES):
            got = int.from_bytes(grp[lane*4:(lane+1)*4], "little")
            if got != w["poke_lanes"][lane]:
                bad += 1
                if bad <= 4:
                    print(f"      word {w['addr']} lane {lane}: "
                          f"{got:#010x} vs {w['poke_lanes'][lane]:#010x}")
    ok &= (bad == 0)
    print(f"  vs proven poke_lanes  : "
          f"{TILE_WORDS*LANES - bad}/{TILE_WORDS*LANES} exact")

    # Re-derive the golden from the PAYLOAD alone, using the documented
    # convention y[j] = sum_i W_loaded[i][j]*a[i]. This catches a
    # transpose or byte-order error that the lane check cannot, because
    # it goes through the arithmetic rather than the representation.
    def s8(b):
        return b - 256 if b > 127 else b

    wl = [[s8(payload[i*WORD_BYTES + j]) for j in range(N)] for i in range(N)]
    ah = int(sorted(ldr.spec["activation_words"],
                    key=lambda x: x["addr"])[0]["hex"], 16)
    av = [s8((ah >> (8*i)) & 0xFF) for i in range(N)]
    y = [sum(wl[i][j]*av[i] for i in range(N)) for j in range(N)]
    g = ldr.spec["expected"]["golden_int32"]
    ok &= (y == g)
    print(f"  golden re-derived     : {'MATCH' if y == g else 'MISMATCH'}")
    if y != g:
        print(f"      got    {y}")
        print(f"      expect {g}")

    align_note = ("source alignment is checked at transfer time "
                  f"({STREAM_ALIGN} B, no DRE)")
    print(f"  note                  : {align_note}")
    print("  SELFCHECK             : " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        raise SystemExit(0 if selfcheck() else 1)
    ldr = DmaWeightLoader()
    raise SystemExit(0 if DmaWeightLoader.report(ldr.run()) else 1)
