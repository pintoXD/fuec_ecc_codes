/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINMISSING */
module tb_fuec_encoder_48_32 (
);
    
    reg [31:0] test_word;
    reg [47:0] code_word;  
    reg [15:0] p;
    fuec_encoder_48_32 enc_dut(.d(test_word), .cw(code_word));


    reg  [47:0] r;
    wire [15:0] s;
    wire [47:0] r_fix;
    wire [31:0] pos_error;
    wire        no_error;
    wire        corrected;
    wire        uncorrectable;

    fuec_decoder_48_32 uut (
        .r(r),
        .s(s),
        .r_fix(r_fix),
        .pos_error(pos_error),
        .no_error(no_error),
        .corrected(corrected),
        .uncorrectable(uncorrectable)
    );


    initial begin
        #5;
        test_word = 32'h87654321;
        #10;
        $display("Input data: %h, Encoded data: %h", test_word, code_word);
        $display("Input data: %h, Encoded data: %0b", test_word, code_word);

        #10;
        r = code_word ^ 48'b00000000010; // Introduce a single-bit error
        #10;
        $display("Received data with single-bit error: %h", r);
        #10;
        $display("Syndrome: %h, Corrected data: %h, No error: %b, Corrected: %b, Uncorrectable: %b, Pos_error: %b", s, r_fix, no_error, corrected, uncorrectable, pos_error);
        $display("Syndrome: %h, Corrected data: %h, No error: %0h, Corrected: %0h, Uncorrectable: %0h, Pos_error: %0h", s, r_fix, no_error, corrected, uncorrectable, pos_error);

    end


endmodule
