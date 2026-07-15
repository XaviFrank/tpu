`timescale 1ns / 1ps
//====================================================================
// tile_buffer: simple-dual-port BRAM tile store
//   write port: TB in v1, DMA/AXI in Phase 3
//   read port : controller FETCH; rd_data valid 1 cycle after rd_en
//   Contract (not enforced in RTL): a tile is never read while
//   being written.
//   BRAM inference is fragile - keep exactly this read idiom:
//   no reset on rd_data/mem, no combinational read, no 2nd read port.
//====================================================================
module tile_buffer #(
    parameter int DEPTH      = 64,              // beat-words; 16 tiles at K=4
    parameter int WORD_WIDTH = 32               // one beat: 4 packed int8s
)(
    input  logic                          clk,
    // write port (v1: driven by TB; Phase 3: DMA/AXI attaches here)
    input  logic                          wr_en,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr,
    input  logic [WORD_WIDTH-1:0]         wr_data,
    // read port (controller FETCH)
    input  logic                          rd_en,
    input  logic [$clog2(DEPTH)-1:0]      rd_addr,
    output logic [WORD_WIDTH-1:0]         rd_data   // valid 1 cycle after rd_en
);
    (* ram_style = "block" *)
    logic [WORD_WIDTH-1:0] mem [DEPTH];

    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        if (rd_en) rd_data <= mem[rd_addr];
    end
endmodule
