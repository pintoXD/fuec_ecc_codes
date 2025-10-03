/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
module fuec_decoder_interface (
    input wire [7:0] data,
    input wire [3:0] redundancy,
    output wire [7:0] data_dec,
    output wire [7:0] pos_error
);
    
    wire [11:0] data_in;
    wire [3:0]  syndrome_out;
    wire [11:0] decoder_data_out;

    fuec_decoder_12_8 fuec_decoder (
        .r(data_in),
        .s(syndrome_out), // Syndrome output
        .r_fix(decoder_data_out), // Corrected codeword output
        .no_error(), // No error flag
        .corrected(), // Corrected data output
        .uncorrectable() // Uncorrectable error flag
    );

    assign data_in = {redundancy, data};

    // Selectors for correctable patterns (XNOR equality: &(s ^~ SVAL))
    localparam [3:0] SVAL_sel_e_7 = 4'b0011; 
    localparam [3:0] SVAL_sel_e_4 = 4'b0101;
    localparam [3:0] SVAL_sel_e_0 = 4'b0111; 
    localparam [3:0] SVAL_sel_e_6 = 4'b1010; 
    localparam [3:0] SVAL_sel_e_3 = 4'b1100; 
    localparam [3:0] SVAL_sel_e_2 = 4'b1101; 
    localparam [3:0] SVAL_sel_e_5 = 4'b1110; 
    localparam [3:0] SVAL_sel_e_1 = 4'b1111; 

    assign pos_error[0] = &(syndrome_out ^~ SVAL_sel_e_0);  // e=(0,)
    assign pos_error[1] = &(syndrome_out ^~ SVAL_sel_e_1);  // e=(1,)
    assign pos_error[2] = &(syndrome_out ^~ SVAL_sel_e_2);  // e=(2,)
    assign pos_error[3] = &(syndrome_out ^~ SVAL_sel_e_3);  // e=(3,)
    assign pos_error[4] = &(syndrome_out ^~ SVAL_sel_e_4);  // e=(4,)
    assign pos_error[5] = &(syndrome_out ^~ SVAL_sel_e_5);  // e=(5,)
    assign pos_error[6] = &(syndrome_out ^~ SVAL_sel_e_6);  // e=(6,)
    assign pos_error[7] = &(syndrome_out ^~ SVAL_sel_e_7);  // e=(7,)


    assign data_dec = decoder_data_out[7:0];






endmodule
