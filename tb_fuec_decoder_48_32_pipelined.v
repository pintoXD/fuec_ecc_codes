`timescale 1ns/1ps

module tb_fuec_decoder_48_32_pipelined;
    reg clk;
    reg rst;
    reg [47:0] r;
    wire [15:0] s;
    wire [47:0] r_fix;
    wire [47:0] pos_error;
    wire no_error;
    wire corrected;
    wire uncorrectable;

    // Instantiate the DUT
    fuec_decoder_48_32 dut (
        .clk(clk),
        .rst(rst),
        .r(r),
        .s(s),
        .r_fix(r_fix),
        .pos_error(pos_error),
        .no_error(no_error),
        .corrected(corrected),
        .uncorrectable(uncorrectable)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // Test vectors
    reg [47:0] test_vectors [0:3];
    integer i;

    initial begin
        // Example test vectors (can be extended)
        test_vectors[0] = 48'h000000000000; // all zeros
        test_vectors[1] = 48'h000000000001; // single bit error at LSB
        test_vectors[2] = 48'h000000000002; // single bit error at bit 1
        test_vectors[3] = 48'hFFFFFFFFFFFF; // all ones

        rst = 1;
        r = 0;
        #20;
        rst = 0;

        for (i = 0; i < 4; i = i + 1) begin
            r = test_vectors[i];
            #20; // Wait for pipeline to settle
            $display("Test %0d: r = %h", i, r);
            $display("  s = %h", s);
            $display("  r_fix = %h", r_fix);
            $display("  pos_error = %h", pos_error);
            $display("  no_error = %b, corrected = %b, uncorrectable = %b", no_error, corrected, uncorrectable);
        end

        $display("Testbench finished.");
        $finish;
    end
endmodule
