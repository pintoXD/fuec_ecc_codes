/* verilator lint_off PINMISSING */
/* verilator lint_off UNUSEDSIGNAL */

`timescale 1ns/1ps
// Auto-generated exhaustive decoder testbench
module tb_fuec_decoder_48_32_vectors;
  localparam int N = 48;
  localparam int R = 16;
  localparam int NUM_PATTERNS = 367;

  // DUT signals
  reg  [N-1:0] r;
  wire [R-1:0] s;
  wire [N-1:0] r_fix;
  wire no_error, corrected, uncorrectable;

  reg [31:0] test_word;
  reg [47:0] code_word;

  fuec_encoder_48_32 enc_dut(.d(test_word), .cw(code_word));
  fuec_decoder_48_32 dut(.r(r), .s(s), .r_fix(r_fix), .no_error(no_error), .corrected(corrected), .uncorrectable(uncorrectable));

  // Patterns (bitmasks) derived from selector comments in decoder
  logic [N-1:0] patterns [0:NUM_PATTERNS-1];
  initial begin
    patterns[0] = 48'h000000000001; // e=0
    patterns[1] = 48'h000000000002; // e=1
    patterns[2] = 48'h000000000004; // e=2
    patterns[3] = 48'h000000000008; // e=3
    patterns[4] = 48'h000000000010; // e=4
    patterns[5] = 48'h000000000020; // e=5
    patterns[6] = 48'h000000000040; // e=6
    patterns[7] = 48'h000000000080; // e=7
    patterns[8] = 48'h000000000100; // e=8
    patterns[9] = 48'h000000000200; // e=9
    patterns[10] = 48'h000000000400; // e=10
    patterns[11] = 48'h000000000800; // e=11
    patterns[12] = 48'h000000001000; // e=12
    patterns[13] = 48'h000000002000; // e=13
    patterns[14] = 48'h000000004000; // e=14
    patterns[15] = 48'h000000008000; // e=15
    patterns[16] = 48'h000000010000; // e=16
    patterns[17] = 48'h000000020000; // e=17
    patterns[18] = 48'h000000040000; // e=18
    patterns[19] = 48'h000000080000; // e=19
    patterns[20] = 48'h000000100000; // e=20
    patterns[21] = 48'h000000200000; // e=21
    patterns[22] = 48'h000000400000; // e=22
    patterns[23] = 48'h000000800000; // e=23
    patterns[24] = 48'h000001000000; // e=24
    patterns[25] = 48'h000002000000; // e=25
    patterns[26] = 48'h000004000000; // e=26
    patterns[27] = 48'h000008000000; // e=27
    patterns[28] = 48'h000010000000; // e=28
    patterns[29] = 48'h000020000000; // e=29
    patterns[30] = 48'h000040000000; // e=30
    patterns[31] = 48'h000080000000; // e=31
    patterns[32] = 48'h000100000000; // e=32
    patterns[33] = 48'h000200000000; // e=33
    patterns[34] = 48'h000400000000; // e=34
    patterns[35] = 48'h000800000000; // e=35
    patterns[36] = 48'h001000000000; // e=36
    patterns[37] = 48'h002000000000; // e=37
    patterns[38] = 48'h004000000000; // e=38
    patterns[39] = 48'h008000000000; // e=39
    patterns[40] = 48'h010000000000; // e=40
    patterns[41] = 48'h020000000000; // e=41
    patterns[42] = 48'h040000000000; // e=42
    patterns[43] = 48'h080000000000; // e=43
    patterns[44] = 48'h100000000000; // e=44
    patterns[45] = 48'h200000000000; // e=45
    patterns[46] = 48'h400000000000; // e=46
    patterns[47] = 48'h800000000000; // e=47
    patterns[48] = 48'h000000000003; // e=0,1
    patterns[49] = 48'h000000000005; // e=0,2
    patterns[50] = 48'h000000000009; // e=0,3
    patterns[51] = 48'h000000000006; // e=1,2
    patterns[52] = 48'h00000000000a; // e=1,3
    patterns[53] = 48'h000000000012; // e=1,4
    patterns[54] = 48'h00000000000c; // e=2,3
    patterns[55] = 48'h000000000014; // e=2,4
    patterns[56] = 48'h000000000024; // e=2,5
    patterns[57] = 48'h000000000018; // e=3,4
    patterns[58] = 48'h000000000028; // e=3,5
    patterns[59] = 48'h000000000048; // e=3,6
    patterns[60] = 48'h000000000030; // e=4,5
    patterns[61] = 48'h000000000050; // e=4,6
    patterns[62] = 48'h000000000090; // e=4,7
    patterns[63] = 48'h000000000060; // e=5,6
    patterns[64] = 48'h0000000000a0; // e=5,7
    patterns[65] = 48'h000000000120; // e=5,8
    patterns[66] = 48'h0000000000c0; // e=6,7
    patterns[67] = 48'h000000000140; // e=6,8
    patterns[68] = 48'h000000000240; // e=6,9
    patterns[69] = 48'h000000000180; // e=7,8
    patterns[70] = 48'h000000000280; // e=7,9
    patterns[71] = 48'h000000000480; // e=7,10
    patterns[72] = 48'h000000000300; // e=8,9
    patterns[73] = 48'h000000000500; // e=8,10
    patterns[74] = 48'h000000000900; // e=8,11
    patterns[75] = 48'h000000000600; // e=9,10
    patterns[76] = 48'h000000000a00; // e=9,11
    patterns[77] = 48'h000000001200; // e=9,12
    patterns[78] = 48'h000000000c00; // e=10,11
    patterns[79] = 48'h000000001400; // e=10,12
    patterns[80] = 48'h000000002400; // e=10,13
    patterns[81] = 48'h000000001800; // e=11,12
    patterns[82] = 48'h000000002800; // e=11,13
    patterns[83] = 48'h000000004800; // e=11,14
    patterns[84] = 48'h000000003000; // e=12,13
    patterns[85] = 48'h000000005000; // e=12,14
    patterns[86] = 48'h000000009000; // e=12,15
    patterns[87] = 48'h000000006000; // e=13,14
    patterns[88] = 48'h00000000a000; // e=13,15
    patterns[89] = 48'h000000012000; // e=13,16
    patterns[90] = 48'h00000000c000; // e=14,15
    patterns[91] = 48'h000000014000; // e=14,16
    patterns[92] = 48'h000000024000; // e=14,17
    patterns[93] = 48'h000000018000; // e=15,16
    patterns[94] = 48'h000000028000; // e=15,17
    patterns[95] = 48'h000000048000; // e=15,18
    patterns[96] = 48'h000000030000; // e=16,17
    patterns[97] = 48'h000000050000; // e=16,18
    patterns[98] = 48'h000000090000; // e=16,19
    patterns[99] = 48'h000000060000; // e=17,18
    patterns[100] = 48'h0000000a0000; // e=17,19
    patterns[101] = 48'h000000120000; // e=17,20
    patterns[102] = 48'h0000000c0000; // e=18,19
    patterns[103] = 48'h000000140000; // e=18,20
    patterns[104] = 48'h000000240000; // e=18,21
    patterns[105] = 48'h000000180000; // e=19,20
    patterns[106] = 48'h000000280000; // e=19,21
    patterns[107] = 48'h000000480000; // e=19,22
    patterns[108] = 48'h000000300000; // e=20,21
    patterns[109] = 48'h000000500000; // e=20,22
    patterns[110] = 48'h000000900000; // e=20,23
    patterns[111] = 48'h000000600000; // e=21,22
    patterns[112] = 48'h000000a00000; // e=21,23
    patterns[113] = 48'h000001200000; // e=21,24
    patterns[114] = 48'h000000c00000; // e=22,23
    patterns[115] = 48'h000001400000; // e=22,24
    patterns[116] = 48'h000002400000; // e=22,25
    patterns[117] = 48'h000001800000; // e=23,24
    patterns[118] = 48'h000002800000; // e=23,25
    patterns[119] = 48'h000004800000; // e=23,26
    patterns[120] = 48'h000003000000; // e=24,25
    patterns[121] = 48'h000005000000; // e=24,26
    patterns[122] = 48'h000009000000; // e=24,27
    patterns[123] = 48'h000006000000; // e=25,26
    patterns[124] = 48'h00000a000000; // e=25,27
    patterns[125] = 48'h000012000000; // e=25,28
    patterns[126] = 48'h00000c000000; // e=26,27
    patterns[127] = 48'h000014000000; // e=26,28
    patterns[128] = 48'h000024000000; // e=26,29
    patterns[129] = 48'h000018000000; // e=27,28
    patterns[130] = 48'h000028000000; // e=27,29
    patterns[131] = 48'h000048000000; // e=27,30
    patterns[132] = 48'h000030000000; // e=28,29
    patterns[133] = 48'h000050000000; // e=28,30
    patterns[134] = 48'h000090000000; // e=28,31
    patterns[135] = 48'h000060000000; // e=29,30
    patterns[136] = 48'h0000a0000000; // e=29,31
    patterns[137] = 48'h000120000000; // e=29,32
    patterns[138] = 48'h0000c0000000; // e=30,31
    patterns[139] = 48'h000140000000; // e=30,32
    patterns[140] = 48'h000240000000; // e=30,33
    patterns[141] = 48'h000180000000; // e=31,32
    patterns[142] = 48'h000280000000; // e=31,33
    patterns[143] = 48'h000480000000; // e=31,34
    patterns[144] = 48'h000300000000; // e=32,33
    patterns[145] = 48'h000500000000; // e=32,34
    patterns[146] = 48'h000900000000; // e=32,35
    patterns[147] = 48'h000600000000; // e=33,34
    patterns[148] = 48'h000a00000000; // e=33,35
    patterns[149] = 48'h001200000000; // e=33,36
    patterns[150] = 48'h000c00000000; // e=34,35
    patterns[151] = 48'h001400000000; // e=34,36
    patterns[152] = 48'h002400000000; // e=34,37
    patterns[153] = 48'h001800000000; // e=35,36
    patterns[154] = 48'h002800000000; // e=35,37
    patterns[155] = 48'h004800000000; // e=35,38
    patterns[156] = 48'h003000000000; // e=36,37
    patterns[157] = 48'h005000000000; // e=36,38
    patterns[158] = 48'h009000000000; // e=36,39
    patterns[159] = 48'h006000000000; // e=37,38
    patterns[160] = 48'h00a000000000; // e=37,39
    patterns[161] = 48'h012000000000; // e=37,40
    patterns[162] = 48'h00c000000000; // e=38,39
    patterns[163] = 48'h014000000000; // e=38,40
    patterns[164] = 48'h024000000000; // e=38,41
    patterns[165] = 48'h018000000000; // e=39,40
    patterns[166] = 48'h028000000000; // e=39,41
    patterns[167] = 48'h048000000000; // e=39,42
    patterns[168] = 48'h030000000000; // e=40,41
    patterns[169] = 48'h050000000000; // e=40,42
    patterns[170] = 48'h090000000000; // e=40,43
    patterns[171] = 48'h060000000000; // e=41,42
    patterns[172] = 48'h0a0000000000; // e=41,43
    patterns[173] = 48'h120000000000; // e=41,44
    patterns[174] = 48'h0c0000000000; // e=42,43
    patterns[175] = 48'h140000000000; // e=42,44
    patterns[176] = 48'h240000000000; // e=42,45
    patterns[177] = 48'h180000000000; // e=43,44
    patterns[178] = 48'h280000000000; // e=43,45
    patterns[179] = 48'h480000000000; // e=43,46
    patterns[180] = 48'h300000000000; // e=44,45
    patterns[181] = 48'h500000000000; // e=44,46
    patterns[182] = 48'h900000000000; // e=44,47
    patterns[183] = 48'h600000000000; // e=45,46
    patterns[184] = 48'ha00000000000; // e=45,47
    patterns[185] = 48'hc00000000000; // e=46,47
    patterns[186] = 48'h000000000007; // e=0,1,2
    patterns[187] = 48'h00000000000b; // e=0,1,3
    patterns[188] = 48'h00000000000d; // e=0,2,3
    patterns[189] = 48'h00000000000e; // e=1,2,3
    patterns[190] = 48'h000000000016; // e=1,2,4
    patterns[191] = 48'h00000000001a; // e=1,3,4
    patterns[192] = 48'h00000000001c; // e=2,3,4
    patterns[193] = 48'h00000000002c; // e=2,3,5
    patterns[194] = 48'h000000000034; // e=2,4,5
    patterns[195] = 48'h000000000038; // e=3,4,5
    patterns[196] = 48'h000000000058; // e=3,4,6
    patterns[197] = 48'h000000000068; // e=3,5,6
    patterns[198] = 48'h000000000070; // e=4,5,6
    patterns[199] = 48'h0000000000b0; // e=4,5,7
    patterns[200] = 48'h0000000000d0; // e=4,6,7
    patterns[201] = 48'h0000000000e0; // e=5,6,7
    patterns[202] = 48'h000000000160; // e=5,6,8
    patterns[203] = 48'h0000000001a0; // e=5,7,8
    patterns[204] = 48'h0000000001c0; // e=6,7,8
    patterns[205] = 48'h0000000002c0; // e=6,7,9
    patterns[206] = 48'h000000000340; // e=6,8,9
    patterns[207] = 48'h000000000380; // e=7,8,9
    patterns[208] = 48'h000000000580; // e=7,8,10
    patterns[209] = 48'h000000000680; // e=7,9,10
    patterns[210] = 48'h000000000700; // e=8,9,10
    patterns[211] = 48'h000000000b00; // e=8,9,11
    patterns[212] = 48'h000000000d00; // e=8,10,11
    patterns[213] = 48'h000000000e00; // e=9,10,11
    patterns[214] = 48'h000000001600; // e=9,10,12
    patterns[215] = 48'h000000001a00; // e=9,11,12
    patterns[216] = 48'h000000001c00; // e=10,11,12
    patterns[217] = 48'h000000002c00; // e=10,11,13
    patterns[218] = 48'h000000003400; // e=10,12,13
    patterns[219] = 48'h000000003800; // e=11,12,13
    patterns[220] = 48'h000000005800; // e=11,12,14
    patterns[221] = 48'h000000006800; // e=11,13,14
    patterns[222] = 48'h000000007000; // e=12,13,14
    patterns[223] = 48'h00000000b000; // e=12,13,15
    patterns[224] = 48'h00000000d000; // e=12,14,15
    patterns[225] = 48'h00000000e000; // e=13,14,15
    patterns[226] = 48'h000000016000; // e=13,14,16
    patterns[227] = 48'h00000001a000; // e=13,15,16
    patterns[228] = 48'h00000001c000; // e=14,15,16
    patterns[229] = 48'h00000002c000; // e=14,15,17
    patterns[230] = 48'h000000034000; // e=14,16,17
    patterns[231] = 48'h000000038000; // e=15,16,17
    patterns[232] = 48'h000000058000; // e=15,16,18
    patterns[233] = 48'h000000068000; // e=15,17,18
    patterns[234] = 48'h000000070000; // e=16,17,18
    patterns[235] = 48'h0000000b0000; // e=16,17,19
    patterns[236] = 48'h0000000d0000; // e=16,18,19
    patterns[237] = 48'h0000000e0000; // e=17,18,19
    patterns[238] = 48'h000000160000; // e=17,18,20
    patterns[239] = 48'h0000001a0000; // e=17,19,20
    patterns[240] = 48'h0000001c0000; // e=18,19,20
    patterns[241] = 48'h0000002c0000; // e=18,19,21
    patterns[242] = 48'h000000340000; // e=18,20,21
    patterns[243] = 48'h000000380000; // e=19,20,21
    patterns[244] = 48'h000000580000; // e=19,20,22
    patterns[245] = 48'h000000680000; // e=19,21,22
    patterns[246] = 48'h000000700000; // e=20,21,22
    patterns[247] = 48'h000000b00000; // e=20,21,23
    patterns[248] = 48'h000000d00000; // e=20,22,23
    patterns[249] = 48'h000000e00000; // e=21,22,23
    patterns[250] = 48'h000001600000; // e=21,22,24
    patterns[251] = 48'h000001a00000; // e=21,23,24
    patterns[252] = 48'h000001c00000; // e=22,23,24
    patterns[253] = 48'h000002c00000; // e=22,23,25
    patterns[254] = 48'h000003400000; // e=22,24,25
    patterns[255] = 48'h000003800000; // e=23,24,25
    patterns[256] = 48'h000005800000; // e=23,24,26
    patterns[257] = 48'h000006800000; // e=23,25,26
    patterns[258] = 48'h000007000000; // e=24,25,26
    patterns[259] = 48'h00000b000000; // e=24,25,27
    patterns[260] = 48'h00000d000000; // e=24,26,27
    patterns[261] = 48'h00000e000000; // e=25,26,27
    patterns[262] = 48'h000016000000; // e=25,26,28
    patterns[263] = 48'h00001a000000; // e=25,27,28
    patterns[264] = 48'h00001c000000; // e=26,27,28
    patterns[265] = 48'h00002c000000; // e=26,27,29
    patterns[266] = 48'h000034000000; // e=26,28,29
    patterns[267] = 48'h000038000000; // e=27,28,29
    patterns[268] = 48'h000058000000; // e=27,28,30
    patterns[269] = 48'h000068000000; // e=27,29,30
    patterns[270] = 48'h000070000000; // e=28,29,30
    patterns[271] = 48'h0000b0000000; // e=28,29,31
    patterns[272] = 48'h0000d0000000; // e=28,30,31
    patterns[273] = 48'h0000e0000000; // e=29,30,31
    patterns[274] = 48'h000160000000; // e=29,30,32
    patterns[275] = 48'h0001a0000000; // e=29,31,32
    patterns[276] = 48'h0001c0000000; // e=30,31,32
    patterns[277] = 48'h0002c0000000; // e=30,31,33
    patterns[278] = 48'h000340000000; // e=30,32,33
    patterns[279] = 48'h000380000000; // e=31,32,33
    patterns[280] = 48'h000580000000; // e=31,32,34
    patterns[281] = 48'h000680000000; // e=31,33,34
    patterns[282] = 48'h000700000000; // e=32,33,34
    patterns[283] = 48'h000b00000000; // e=32,33,35
    patterns[284] = 48'h000d00000000; // e=32,34,35
    patterns[285] = 48'h000e00000000; // e=33,34,35
    patterns[286] = 48'h001600000000; // e=33,34,36
    patterns[287] = 48'h001a00000000; // e=33,35,36
    patterns[288] = 48'h001c00000000; // e=34,35,36
    patterns[289] = 48'h002c00000000; // e=34,35,37
    patterns[290] = 48'h003400000000; // e=34,36,37
    patterns[291] = 48'h003800000000; // e=35,36,37
    patterns[292] = 48'h005800000000; // e=35,36,38
    patterns[293] = 48'h006800000000; // e=35,37,38
    patterns[294] = 48'h007000000000; // e=36,37,38
    patterns[295] = 48'h00b000000000; // e=36,37,39
    patterns[296] = 48'h00d000000000; // e=36,38,39
    patterns[297] = 48'h00e000000000; // e=37,38,39
    patterns[298] = 48'h016000000000; // e=37,38,40
    patterns[299] = 48'h01a000000000; // e=37,39,40
    patterns[300] = 48'h01c000000000; // e=38,39,40
    patterns[301] = 48'h02c000000000; // e=38,39,41
    patterns[302] = 48'h034000000000; // e=38,40,41
    patterns[303] = 48'h038000000000; // e=39,40,41
    patterns[304] = 48'h058000000000; // e=39,40,42
    patterns[305] = 48'h068000000000; // e=39,41,42
    patterns[306] = 48'h070000000000; // e=40,41,42
    patterns[307] = 48'h0b0000000000; // e=40,41,43
    patterns[308] = 48'h0d0000000000; // e=40,42,43
    patterns[309] = 48'h0e0000000000; // e=41,42,43
    patterns[310] = 48'h160000000000; // e=41,42,44
    patterns[311] = 48'h1a0000000000; // e=41,43,44
    patterns[312] = 48'h1c0000000000; // e=42,43,44
    patterns[313] = 48'h2c0000000000; // e=42,43,45
    patterns[314] = 48'h340000000000; // e=42,44,45
    patterns[315] = 48'h380000000000; // e=43,44,45
    patterns[316] = 48'h580000000000; // e=43,44,46
    patterns[317] = 48'h680000000000; // e=43,45,46
    patterns[318] = 48'h700000000000; // e=44,45,46
    patterns[319] = 48'hb00000000000; // e=44,45,47
    patterns[320] = 48'hd00000000000; // e=44,46,47
    patterns[321] = 48'he00000000000; // e=45,46,47
    patterns[322] = 48'h00000000000f; // e=0,1,2,3
    patterns[323] = 48'h00000000001e; // e=1,2,3,4
    patterns[324] = 48'h00000000003c; // e=2,3,4,5
    patterns[325] = 48'h000000000078; // e=3,4,5,6
    patterns[326] = 48'h0000000000f0; // e=4,5,6,7
    patterns[327] = 48'h0000000001e0; // e=5,6,7,8
    patterns[328] = 48'h0000000003c0; // e=6,7,8,9
    patterns[329] = 48'h000000000780; // e=7,8,9,10
    patterns[330] = 48'h000000000f00; // e=8,9,10,11
    patterns[331] = 48'h000000001e00; // e=9,10,11,12
    patterns[332] = 48'h000000003c00; // e=10,11,12,13
    patterns[333] = 48'h000000007800; // e=11,12,13,14
    patterns[334] = 48'h00000000f000; // e=12,13,14,15
    patterns[335] = 48'h00000001e000; // e=13,14,15,16
    patterns[336] = 48'h00000003c000; // e=14,15,16,17
    patterns[337] = 48'h000000078000; // e=15,16,17,18
    patterns[338] = 48'h0000000f0000; // e=16,17,18,19
    patterns[339] = 48'h0000001e0000; // e=17,18,19,20
    patterns[340] = 48'h0000003c0000; // e=18,19,20,21
    patterns[341] = 48'h000000780000; // e=19,20,21,22
    patterns[342] = 48'h000000f00000; // e=20,21,22,23
    patterns[343] = 48'h000001e00000; // e=21,22,23,24
    patterns[344] = 48'h000003c00000; // e=22,23,24,25
    patterns[345] = 48'h000007800000; // e=23,24,25,26
    patterns[346] = 48'h00000f000000; // e=24,25,26,27
    patterns[347] = 48'h00001e000000; // e=25,26,27,28
    patterns[348] = 48'h00003c000000; // e=26,27,28,29
    patterns[349] = 48'h000078000000; // e=27,28,29,30
    patterns[350] = 48'h0000f0000000; // e=28,29,30,31
    patterns[351] = 48'h0001e0000000; // e=29,30,31,32
    patterns[352] = 48'h0003c0000000; // e=30,31,32,33
    patterns[353] = 48'h000780000000; // e=31,32,33,34
    patterns[354] = 48'h000f00000000; // e=32,33,34,35
    patterns[355] = 48'h001e00000000; // e=33,34,35,36
    patterns[356] = 48'h003c00000000; // e=34,35,36,37
    patterns[357] = 48'h007800000000; // e=35,36,37,38
    patterns[358] = 48'h00f000000000; // e=36,37,38,39
    patterns[359] = 48'h01e000000000; // e=37,38,39,40
    patterns[360] = 48'h03c000000000; // e=38,39,40,41
    patterns[361] = 48'h078000000000; // e=39,40,41,42
    patterns[362] = 48'h0f0000000000; // e=40,41,42,43
    patterns[363] = 48'h1e0000000000; // e=41,42,43,44
    patterns[364] = 48'h3c0000000000; // e=42,43,44,45
    patterns[365] = 48'h780000000000; // e=43,44,45,46
    patterns[366] = 48'hf00000000000; // e=44,45,46,47
  end

  integer i; integer errors = 0;
  

  task check_pattern(input [N-1:0] mask);
    begin
      r = mask; #1;  // combinational settle
      if (mask == '0) begin
        if (!(no_error && !corrected && !uncorrectable && r_fix == '0)) begin
          $display("[FAIL][zero] r=%b no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b", r, no_error, corrected, uncorrectable, r_fix);
          errors = errors + 1;
        end
      end else begin
        // Expect correction: corrected=1, r_fix back to zero, no_error=0
        // $display("[TEST][mask=%b] r=%b no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%0h", mask, r, no_error, corrected, uncorrectable, r_fix);
        if (!(corrected && !no_error && !uncorrectable && r_fix == '0)) begin
          $display("[FAIL][mask=%b] no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b", mask, no_error, corrected, uncorrectable, r_fix);
          errors = errors + 1;
        end
      end
    end
  endtask
  
  task check_codeword_masking_and_correction(input [N-1:0] mask);
    begin
      r = mask ^ code_word; #1;  // combinational settle
      if (mask == '0) begin
        if (!(no_error && !corrected && !uncorrectable && r_fix == code_word)) begin
          $display("[FAIL][zero] r=%b no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b", r, no_error, corrected, uncorrectable, r_fix);
          errors = errors + 1;
        end
      end else begin
        // Expect correction: corrected=1, r_fix back to zero, no_error=0
        $display("[TEST][mask=%b] r=%0h no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%0h", mask, r, no_error, corrected, uncorrectable, r_fix);
        if (!(corrected && !no_error && !uncorrectable && r_fix == code_word)) begin
          $display("[FAIL][mask=%b] no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b", mask, no_error, corrected, uncorrectable, r_fix);
          errors = errors + 1;
        end
      end
    end
  endtask

  
  initial begin
    // Zero vector sanity
    check_pattern('0);
    
    // All listed correctable patterns
    for (i = 0; i < NUM_PATTERNS; i = i + 1) begin
      check_pattern(patterns[i]);
    end
    if (errors == 0) $display("All %0d patterns PASSED.", NUM_PATTERNS); else $display("%0d patterns FAILED.", errors);
    
    
    $display("----- Now testing codeword masking and correction -----");
    errors = 0;
    assign test_word = 32'hFEC1918A;
    #30;
    for (i = 0; i < NUM_PATTERNS; i = i + 1) begin
      check_codeword_masking_and_correction(patterns[i]);
    end
    if (errors == 0) $display("All %0d patterns PASSED.", NUM_PATTERNS); else $display("%0d patterns FAILED.", errors);

    #30;

    $display("----- Now testing single case -----");

    r = 48'habf600120212;
    #10;
    $display("r=%0h no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%0h", r, no_error, corrected, uncorrectable, r_fix);
    #10;


    $finish;
  end
endmodule

