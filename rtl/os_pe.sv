`timescale 1ns / 1ps
//====================================================================
// pe_os: Output-stationary processing element
//   - Instantiates mac_unit for all arithmetic
//   - Adds systolic forwarding: a -> right, b -> down, 1 reg per hop
//====================================================================
module pe_os #( 
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          clear,
    input  logic                          drain,          // NEW
    input  logic signed [ACC_WIDTH-1:0]   acc_in_above,   // NEW
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  a_in,
    input  logic signed [DATA_WIDTH-1:0]  b_in,

    output logic                          valid_out,
    output logic signed [DATA_WIDTH-1:0]  a_out,
    output logic signed [DATA_WIDTH-1:0]  b_out,
    output logic signed [ACC_WIDTH-1:0]   acc_out,
    output logic                          acc_valid
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= valid_in;
    end

    always_ff @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
    end

    mac_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear     (clear),
        .drain     (drain),
        .drain_in  (acc_in_above),
        .valid_in  (valid_in),
        .a_in      (a_in),
        .b_in      (b_in),
        .acc_out   (acc_out),
        .valid_out (acc_valid)
    );

endmodule