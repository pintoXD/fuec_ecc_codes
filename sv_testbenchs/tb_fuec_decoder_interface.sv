/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off PINMISSING */
/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNDRIVEN */

`timescale 1ns/1ps

// Testbench for fuec_decoder_interface
// Verifies that single-bit errors on data positions (0..7) produce the expected
// one-hot pos_error, and that parity-only (positions 8..11) do not assert pos_error.
module tb_fuec_decoder_interface;
    logic [7:0] data_in;
    logic [3:0] parity_bits_out;
    logic [11:0] encoded_data_out;
    logic [11:0] dirty_encoded_data_out;
    logic [7:0] decoder_data_out;
    logic [7:0] corrected_data_out;
    logic [7:0] pos_error;
    logic [3:0] syndrome_out;
    logic no_error_flag;
    logic corrected_flag;
    logic uncorrectable_flag;


    fuec_encoder_12_8 fuec_encoder (
        .d(data_in),
        .p(parity_bits_out),
        .cw(encoded_data_out)
    );

    fuec_decoder_interface fuec_decoder(
        .data(dirty_encoded_data_out[7:0]),
        .redundancy(dirty_encoded_data_out[11:8]),
        .data_dec(decoder_data_out),
        .pos_error(pos_error)
    );


    int counter;
    int indexer;

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

        $display("Test case 1 with 1-bit error...");
        dirty_encoded_data_out = encoded_data_out;
        dirty_encoded_data_out[0] = ~dirty_encoded_data_out[0]; // Introduce an error

        #10; // Wait for some time to allow processing
        $display("Dirty Encoded Data (with error): %b", dirty_encoded_data_out);
        // $display("Syndrome: %b", syndrome_out);
        $display("Corrected Codeword: %b", decoder_data_out);
        $display("Pos Error: [%d]", pos_error - 1);
        // $display("No Error Flag: %b", no_error_flag);
        // $display("Corrected Flag: %b", corrected_flag);
        $display("Uncorrectable Flag: %b", uncorrectable_flag);
        #10;


        corrected_data_out = decoder_data_out[7:0]; // Extract corrected data bits
        $display("Corrected Data: %b", corrected_data_out == data_in ? 1'b1 : 1'b0);
        /*
        $display("\n\n ------------------------------- \n\n");

        $display("Test case 2 with 2 adjacent bits error...");
        dirty_encoded_data_out = encoded_data_out;
        dirty_encoded_data_out[4] = ~dirty_encoded_data_out[4]; // Introduce an error
        dirty_encoded_data_out[5] = ~dirty_encoded_data_out[5]; // Introduce an error

        #10; // Wait for some time to allow processing
        $display("Encoded Data:                    %b", encoded_data_out);
        $display("Dirty Encoded Data (with error): %b", dirty_encoded_data_out);
        $display("Syndrome:                        %b", syndrome_out);
        $display("Corrected Codeword:              %b", decoder_data_out);
        $display("No Error Flag:                   %b", no_error_flag);
        $display("Corrected Flag:                  %b", corrected_flag);
        $display("Uncorrectable Flag:              %b", uncorrectable_flag);
        #10;


        corrected_data_out = decoder_data_out[7:0]; // Extract corrected data bits
        $display("Corrected Data: %b", corrected_data_out == data_in);

        $display("\n\n ------------------------------- \n\n");

        $display("Test case 3 with 2 random bits error...");
        dirty_encoded_data_out = encoded_data_out;
        dirty_encoded_data_out[0] = ~dirty_encoded_data_out[0]; // Introduce an error
        dirty_encoded_data_out[7] = ~dirty_encoded_data_out[7]; // Introduce an error

        #10; // Wait for some time to allow processing
        $display("Encoded Data:                    %b", encoded_data_out);
        $display("Dirty Encoded Data (with error): %b", dirty_encoded_data_out);
        $display("Syndrome:                        %b", syndrome_out);
        $display("Corrected Codeword:              %b", decoder_data_out);
        $display("No Error Flag:                   %b", no_error_flag);
        $display("Corrected Flag:                  %b", corrected_flag);
        $display("Uncorrectable Flag:              %b", uncorrectable_flag);
        #10;


        corrected_data_out = decoder_data_out[7:0]; // Extract corrected data bits
        $display("Corrected Data: %b", corrected_data_out == data_in);

        */

        $display("\n\n ------------------------------- \n\n");
        $display("Test case with all single-bit errors...");
        
        counter = 0;
        indexer = 8;
        for (int i = 0; i < indexer; i++) begin
            dirty_encoded_data_out = encoded_data_out;
            dirty_encoded_data_out[i] = ~dirty_encoded_data_out[i]; // Introduce an error
            #5;
            if (encoded_data_out[7:0] == decoder_data_out) begin
                // $display("Error at position %0d corrected successfully.", i);
                $display("Pos Error %d: [%b] corrected", i, pos_error);
                counter++;
            end else begin
                $display("Error at position %0d NOT corrected.", i);
            end
            #10;
        end

        $display("Single-bit error correction success rate: %0d/%0d", counter, indexer);




    end

endmodule
