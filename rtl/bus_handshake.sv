`timescale 1ns / 1ps
//====================================================================
// bus_handshake: multi-bit value crossing via toggle req/ack.
//   NEVER synchronize the bits of a bus independently - different
//   bits resolve on different edges and the destination can see a
//   torn value. Here the source captures src_data into a holding
//   register and toggles req in the SAME source-domain edge; the
//   destination synchronizes req, and only on the synchronized edge
//   samples the holding register (guaranteed stable: the source
//   ignores src_send while src_busy, i.e. until the ack returns).
//   Round trip is ~3 dst + ~3 src cycles; callers queue re-sends
//   behind src_busy (see axi_lite_ctrl's dirty-flag pattern).
//
// RESET REQUIREMENT: both domains' resets must be asserted/released
// together (one reset controller). A one-sided reset can strand
// req_t/ack_t on opposite phases, wedging src_busy at 1 and
// permanently blocking the channel. This design assumes the SoC
// sequences both resets; there is deliberately no self-recovery.
//====================================================================
module bus_handshake #(
    parameter int W = 32
)(
    input  logic         src_clk,
    input  logic         src_rst_n,
    input  logic [W-1:0] src_data,
    input  logic         src_send,     // 1-cycle pulse; ignored while src_busy
    output logic         src_busy,
    input  logic         dst_clk,
    input  logic         dst_rst_n,
    output logic [W-1:0] dst_data,     // registered; holds last delivered value
    output logic         dst_valid     // 1-cycle pulse per delivery
);

    //----------------------------------------------------------------
    // Source: capture + toggle request
    //----------------------------------------------------------------
    logic [W-1:0] hold_q;
    logic         req_t;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            hold_q <= '0;
            req_t  <= 1'b0;
        end else if (src_send && !src_busy) begin
            hold_q <= src_data;
            req_t  <= ~req_t;
        end
    end

    //----------------------------------------------------------------
    // Destination: sync request, capture on edge, toggle ack
    //----------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic req_meta;
    (* ASYNC_REG = "TRUE" *) logic req_s1;
    logic req_s2;
    logic ack_t;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            req_meta  <= 1'b0;
            req_s1    <= 1'b0;
            req_s2    <= 1'b0;
            ack_t     <= 1'b0;
            dst_data  <= '0;
            dst_valid <= 1'b0;
        end else begin
            req_meta  <= req_t;
            req_s1    <= req_meta;
            req_s2    <= req_s1;
            dst_valid <= 1'b0;
            if (req_s1 != req_s2) begin
                dst_data  <= hold_q;   // stable: source blocked until ack
                dst_valid <= 1'b1;
                ack_t     <= ~ack_t;
            end
        end
    end

    //----------------------------------------------------------------
    // Source: sync ack; busy while a request is outstanding
    //----------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic ack_meta;
    (* ASYNC_REG = "TRUE" *) logic ack_s1;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            ack_meta <= 1'b0;
            ack_s1   <= 1'b0;
        end else begin
            ack_meta <= ack_t;
            ack_s1   <= ack_meta;
        end
    end

    assign src_busy = (req_t != ack_s1);

endmodule
