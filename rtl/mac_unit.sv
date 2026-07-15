`timescale 1ns / 1ps
//====================================================================
// mac_unit: 2-stage pipelined multiply-accumulate
//   Stage 1: register a*b product
//   Stage 2: accumulate (clear has priority for tile boundaries)
//====================================================================
(* use_dsp = "yes" *) // necessary to cut down on LUTs
module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          clear,
    input  logic                          drain,      // NEW: shift mode
    input  logic signed [ACC_WIDTH-1:0]   drain_in,   // NEW: from PE above
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  a_in,
    input  logic signed [DATA_WIDTH-1:0]  b_in,
    output logic signed [ACC_WIDTH-1:0]   acc_out,
    output logic                          valid_out
);

    logic signed [2*DATA_WIDTH-1:0] product;
    logic                           valid_pipe;

    // Control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 1'b0;
            valid_out  <= 1'b0;
        end else begin
            valid_pipe <= valid_in;
            valid_out  <= valid_pipe && !clear && !drain;
        end
    end

    // Datapath: priority clear > drain > accumulate
    always_ff @(posedge clk) begin
        product <= a_in * b_in;
        if (clear)
            acc_out <= '0;
        else if (drain)
            acc_out <= drain_in;
        else if (valid_pipe)
            acc_out <= acc_out + product;
    end

endmodule