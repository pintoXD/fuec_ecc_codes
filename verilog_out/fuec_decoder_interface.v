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

    localparam [3:0] SVAL_sel_e_8 = 4'b0001; 
    localparam [3:0] SVAL_sel_e_9 = 4'b0010;  
    localparam [3:0] SVAL_sel_e_10 = 4'b0100; 
    localparam [3:0] SVAL_sel_e_0 = 4'b0110;  
    localparam [3:0] SVAL_sel_e_2 = 4'b0111;  
    localparam [3:0] SVAL_sel_e_11 = 4'b1000; 
    localparam [3:0] SVAL_sel_e_5 = 4'b1001;  
    localparam [3:0] SVAL_sel_e_3 = 4'b1010;  
    localparam [3:0] SVAL_sel_e_6 = 4'b1011;  
    localparam [3:0] SVAL_sel_e_4 = 4'b1101;  
    localparam [3:0] SVAL_sel_e_7 = 4'b1110;  
    localparam [3:0] SVAL_sel_e_1 = 4'b1111;  


    assign pos_error[0] = &(syndrome_out ^~ SVAL_sel_e_0);
    assign pos_error[1] = &(syndrome_out ^~ SVAL_sel_e_1);
    assign pos_error[2] = &(syndrome_out ^~ SVAL_sel_e_2);
    assign pos_error[3] = &(syndrome_out ^~ SVAL_sel_e_3);
    assign pos_error[4] = &(syndrome_out ^~ SVAL_sel_e_4);
    assign pos_error[5] = &(syndrome_out ^~ SVAL_sel_e_5);
    assign pos_error[6] = &(syndrome_out ^~ SVAL_sel_e_6);
    assign pos_error[7] = &(syndrome_out ^~ SVAL_sel_e_7);
    // assign pos_error[8] = &(syndrome_out ^~ SVAL_sel_e_8);
    // assign pos_error[9] = &(syndrome_out ^~ SVAL_sel_e_9);
    // assign pos_error[10] = &(syndrome_out ^~ SVAL_sel_e_10);
    // assign pos_error[11] = &(syndrome_out ^~ SVAL_sel_e_11);

    assign data_dec = decoder_data_out[7:0];






endmodule
