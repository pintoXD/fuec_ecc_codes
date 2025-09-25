`timescale 1ns/1ps
module fuec_13_8_tb ();

    logic [7:0] data_in;
    logic [4:0] parity_bits_out;
    logic [12:0] encoded_data_out;
    logic [12:0] dirty_encoded_data_out;
    logic [12:0] dirty_encoded_data_out;
    logic [7:0] decoded_data_out;
    logic 


    fuec_encoder_13_8 fuec_encoder (
        .d(data_in),
        .p(parity_bits_out),
        .cw(encoded_data_out)
    );

    fuec_decoder_13_8 fuec_decoder (
        .r(dirty_encoded_data_out),
        .s(), // Syndrome output
        .r_fix(), // Corrected codeword output
        .no_error(), // No error flag
        .corrected(), // Corrected data output
        .uncorrectable() // Uncorrectable error flag
    );


    initial begin
        #5;
        $display("Begining the encoder tests...");
        #5;
        data_in = 8'b10101100; // Example input data
        #10; // Wait for some time to allow processing
        $display("Input Data: %b", data_in);
        $display("Parity Bits: %b", parity_bits_out);
        $display("Encoded Data: %b", encoded_data_out);
        #10;



        



    end


    
endmodule
