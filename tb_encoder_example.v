`timescale 1ns/1ps
module tb_fuec_encoder_12_4;
  reg  [7:0] d;
  wire [3:0] p;
  wire [11:0] cw;

  fuec_encoder_12_4 uut(.d(d), .p(p), .cw(cw));

  integer i; integer errors = 0;
  reg [7:0] D [0:7];
  reg [3:0] Pexp [0:7];
  reg [11:0] CWexp [0:7];

  initial begin
    D[0] = 8'h00;
    Pexp[0] = 4'h0;
    CWexp[0] = 12'h000;
    D[1] = 8'h01;
    Pexp[1] = 4'h3;
    CWexp[1] = 12'h301;
    D[2] = 8'ha7;
    Pexp[2] = 4'h8;
    CWexp[2] = 12'h8a7;
    D[3] = 8'hb2;
    Pexp[3] = 4'h7;
    CWexp[3] = 12'h7b2;
    D[4] = 8'hac;
    Pexp[4] = 4'hd;
    CWexp[4] = 12'hdac;
    D[5] = 8'h3a;
    Pexp[5] = 4'h4;
    CWexp[5] = 12'h43a;
    D[6] = 8'h70;
    Pexp[6] = 4'h4;
    CWexp[6] = 12'h470;
    D[7] = 8'hec;
    Pexp[7] = 4'hb;
    CWexp[7] = 12'hbec;
    #1;
    for (i = 0; i < $size(D); i = i + 1) begin
      d = D[i]; #1;
      if (p !== Pexp[i]) begin
        $display("[FAIL] i=%0d d=%b p=%b exp=%b", i, d, p, Pexp[i]); errors = errors + 1;
      end
      if (cw !== CWexp[i]) begin
        $display("[FAIL] i=%0d d=%b cw=%b exp=%b", i, d, cw, CWexp[i]); errors = errors + 1;
      end
    end
    if (errors == 0) $display("All %0d tests PASSED.", $size(D)); else $display("%0d tests FAILED.", errors);
    $finish;
  end
endmodule