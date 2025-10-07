module fuec_decoder_48_32_interface (
    input wire [31:0] data,
    input wire [15:0] redundancy,
    output wire [31:0] data_dec,
    output wire [31:0] pos_error
);


    wire [47:0] data_in;
    wire [15:0] syndrome_out;
    wire [47:0] decoder_data_out;

    assign data_in = {redundancy, data};
    assign data_dec = decoder_data_out[31:0];


    fuec_decoder_48_32 fuec_decoder (
        .r(data_in),
        .s(syndrome_out), // Syndrome output
        .r_fix(decoder_data_out), // Corrected codeword output
        .pos_error(pos_error), // Position of errors
        .no_error(), // No error flag
        .corrected(), // Corrected data output
        .uncorrectable() // Uncorrectable error flag
    );
endmodule