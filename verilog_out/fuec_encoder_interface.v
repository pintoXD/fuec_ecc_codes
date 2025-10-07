module fuec_encoder_interface (
    input wire [31:0] data,
    output wire [15:0] redundancy
);
    
    fuec_encoder_48_32 fuec_encoder (
        .d(data),
        .p(redundancy)
        .cw()
    );
    
endmodule