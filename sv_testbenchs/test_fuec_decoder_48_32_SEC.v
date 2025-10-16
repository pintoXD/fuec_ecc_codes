/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDSIGNAL */
`timescale 1ns / 1ps

module test_fuec_decoder_48_32_SEC;

    reg  [47:0] r;
    wire [15:0] s;
    wire [47:0] r_fix;
    wire        no_error;
    wire        corrected;
    wire        uncorrectable;

    fuec_decoder_48_32 uut (
        .r(r),
        .s(s),
        .r_fix(r_fix),
        .no_error(no_error),
        .corrected(corrected),
        .uncorrectable(uncorrectable)
    );

    // Helper to flip a single bit
    function [47:0] flip_bit;
        input [47:0] vec;
        input integer idx;
        begin
            flip_bit = vec ^ (48'b1 << idx);
        end
    endfunction

    // Helper to flip two adjacent bits
    function [47:0] flip_adjacent;
        input [47:0] vec;
        input integer idx;
        begin
            flip_adjacent = vec ^ ((48'b11) << idx);
        end
    endfunction

    integer i;

    initial begin
        $display("Starting fuec_decoder_48_32_SEC tests...");

        // Test 1: No error
        r = 48'h0;
        #1;
        if (!no_error) $display("FAIL: no_error should be high for zero input");
        if (corrected) $display("FAIL: corrected should be low for zero input");
        if (uncorrectable) $display("FAIL: uncorrectable should be low for zero input");

        // Test 2: Single-bit error correction (try all bits)
        

        for (i = 0; i < 48; i = i + 1) begin
            r = flip_bit(48'h0, i);
            #10;
            if (corrected !== 1'b1)
                $display("FAIL: corrected should be high for single-bit error at bit %0d", i);
            if (uncorrectable !== 1'b0)
                $display("FAIL: uncorrectable should be low for single-bit error at bit %0d", i);
            if (r_fix !== 48'h0)
                $display("FAIL: r_fix should correct single-bit error at bit %0d", i);
        end

        // Test 3: Double-adjacent error (try bits 0..46)
        for (i = 0; i < 47; i = i + 1) begin
            r = flip_adjacent(48'h0, i);
            #10;
            // For SEC code, only single-bit errors are corrected, double-adjacent may not be corrected
            if (corrected !== 1'b0 && r_fix == flip_adjacent(48'h0, i))
                $display("WARN: Double-adjacent error at bits %0d,%0d not corrected (expected for SEC)", i, i+1);
        end

        // Test 4: Random multi-bit error (uncorrectable)
        r = 48'hF0F0F0F0F0F0;
        #10;
        if (uncorrectable !== 1'b1)
            $display("FAIL: uncorrectable should be high for random multi-bit error");
        if (corrected !== 1'b0)
            $display("FAIL: corrected should be low for random multi-bit error");

        // Test 5: All bits set (uncorrectable)
        r = 48'hFFFFFFFFFFFF;
        #10;
        if (uncorrectable !== 1'b1)
            $display("FAIL: uncorrectable should be high for all bits set");
        if (corrected !== 1'b0)
            $display("FAIL: corrected should be low for all bits set");

        $display("fuec_decoder_48_32_SEC tests completed.");
        $finish;
    end

endmodule
