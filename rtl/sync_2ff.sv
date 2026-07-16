`timescale 1ns / 1ps
//====================================================================
// sync_2ff: 2-flop level synchronizer, single bit. For quasi-static
// levels only (busy) - never for pulses or multi-bit values.
//====================================================================
module sync_2ff (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);

    (* ASYNC_REG = "TRUE" *) logic meta;
    (* ASYNC_REG = "TRUE" *) logic sync1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            meta  <= 1'b0;
            sync1 <= 1'b0;
        end else begin
            meta  <= d;
            sync1 <= meta;
        end
    end

    assign q = sync1;

endmodule
