`timescale 1ns/1ps
module fuec_13_8_tb ();

    logic [7:0] data_in;
    logic [4:0] parity_bits_out;
    logic [12:0] encoded_data_out;
    logic [12:0] dirty_encoded_data_out;
    logic [12:0] decoder_data_out;
    logic [7:0] corrected_data_out;
    logic [4:0] syndrome_out;
    logic no_error_flag;
    logic corrected_flag;
    logic uncorrectable_flag;


    fuec_encoder_13_8 fuec_encoder (
        .d(data_in),
        .p(parity_bits_out),
        .cw(encoded_data_out)
    );

    fuec_decoder_13_8 fuec_decoder (
        .r(dirty_encoded_data_out),
        .s(syndrome_out), // Syndrome output
        .r_fix(decoder_data_out), // Corrected codeword output
        .no_error(no_error_flag), // No error flag
        .corrected(corrected_flag), // Corrected data output
        .uncorrectable(uncorrectable_flag) // Uncorrectable error flag
    );

    int counter;

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
        $display("Syndrome: %b", syndrome_out);
        $display("Corrected Codeword: %b", decoder_data_out);
        $display("No Error Flag: %b", no_error_flag);
        $display("Corrected Flag: %b", corrected_flag);
        $display("Uncorrectable Flag: %b", uncorrectable_flag);
        #10;


        corrected_data_out = decoder_data_out[7:0]; // Extract corrected data bits
        $display("Corrected Data: %b", corrected_data_out == data_in);
        
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


        $display("\n\n ------------------------------- \n\n");
        $display("Test case 4 with all single-bit errors...");
        
        counter = 0;
        for (int i = 0; i < 13; i++) begin
            dirty_encoded_data_out = encoded_data_out;
            dirty_encoded_data_out[i] = ~dirty_encoded_data_out[i]; // Introduce an error
            #5;
            if (encoded_data_out == decoder_data_out) begin
                // $display("Error at position %0d corrected successfully.", i);
                counter++;
            end else begin
                $display("Error at position %0d NOT corrected.", i);
            end
            #10;
        end

        $display("Single-bit error correction success rate: %0d/13", counter);





    end


    
endmodule
