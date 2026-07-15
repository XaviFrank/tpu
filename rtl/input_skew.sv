`timescale 1ns / 1ps
//====================================================================
// input_skew: delays stream i by (i+1) cycles
//   Relative skew between streams is i - the systolic parallelogram.
//   The +1 makes EVERY path registered: uniform timing contract,
//   no combinational feedthrough, no special cases.
//====================================================================
module input_skew #(
    parameter int N          = 4,
    parameter int DATA_WIDTH = 8
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in  [N],
    input  logic signed [DATA_WIDTH-1:0]  data_in   [N],
    output logic                          valid_out [N],
    output logic signed [DATA_WIDTH-1:0]  data_out  [N]
);

    for (genvar i = 0; i < N; i++) begin : gen_delay

        localparam int DEPTH = i + 1;

        logic signed [DATA_WIDTH-1:0] pipe_d [DEPTH];
        logic                         pipe_v [DEPTH];

        // Datapath: no reset (valid gates everything downstream)
        always_ff @(posedge clk) begin
            pipe_d[0] <= data_in[i];
            for (int k = 1; k < DEPTH; k++)
                pipe_d[k] <= pipe_d[k-1];
        end

        // Control: reset, so the array wakes up quiet
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int k = 0; k < DEPTH; k++)
                    pipe_v[k] <= 1'b0;
            end else begin
                pipe_v[0] <= valid_in[i];
                for (int k = 1; k < DEPTH; k++)
                    pipe_v[k] <= pipe_v[k-1];
            end
        end

        assign data_out[i]  = pipe_d[DEPTH-1];
        assign valid_out[i] = pipe_v[DEPTH-1];

    end

endmodule