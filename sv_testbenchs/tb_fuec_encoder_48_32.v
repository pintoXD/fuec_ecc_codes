/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINMISSING */
module tb_fuec_encoder_48_32 (
);
    
    reg [31:0] test_word;
    reg [47:0] code_word;  
    reg [15:0] p;
    fuec_encoder_48_32 enc_dut(.d(test_word), .cw(code_word));

    initial begin
        #5;
        test_word = 32'h87654321;
        #10;
        $display("Input data: %h, Encoded data: %h", test_word, code_word);
        $display("Input data: %h, Encoded data: %0b", test_word, code_word);

    end


endmodule
