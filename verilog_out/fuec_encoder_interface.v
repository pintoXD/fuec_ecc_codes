module fuec_encoder_interface (
    input wire [7:0] data,
    output wire [3:0] redundancy
);
    
    fuec_encoder_12_8 fuec_encoder (
        .d(data),
        .p(redundancy)
        .cw()
    );
    
endmodule