`timescale 1ns / 1ps
//====================================================================
// pulse_sync: single-cycle pulse crossing via toggle-and-synchronize.
// Source toggles a flop per pulse; destination synchronizes the
// toggle and edge-detects it. Source pulses must be spaced more than
// ~3 destination cycles apart or toggles can be swallowed (AXI
// transaction pacing guarantees that for go/done here).
//====================================================================
module pulse_sync (
    input  logic src_clk,
    input  logic src_rst_n,
    input  logic src_pulse,
    input  logic dst_clk,
    input  logic dst_rst_n,
    output logic dst_pulse
);

    logic req_t;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n)
            req_t <= 1'b0;
        else if (src_pulse)
            req_t <= ~req_t;
    end

    (* ASYNC_REG = "TRUE" *) logic meta;
    (* ASYNC_REG = "TRUE" *) logic sync1;
    logic sync2;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            meta  <= 1'b0;
            sync1 <= 1'b0;
            sync2 <= 1'b0;
        end else begin
            meta  <= req_t;
            sync1 <= meta;
            sync2 <= sync1;
        end
    end

    assign dst_pulse = sync1 ^ sync2;

endmodule
