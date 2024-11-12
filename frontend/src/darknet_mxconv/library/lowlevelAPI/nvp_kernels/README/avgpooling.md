About
 - This kernel is for general avgpooling
 - div_multiplier and div_shifter is for division 1/{kernel_size} 
   e.g.) if you want 1/2 div_multiplier = 1, div_shifter = 1
   e.g.) if you want 1/3, approximately transform 1/3 into div_multiplier / 2^{div_shifter} form
 - ronuding is done by kernel 

parameter list
sm[00] = in_address0
sm[01] = in_width0
sm[02] = in_address1
sm[03] = in_width1
sm[04] = in_height0
sm[05] = in_height1
sm[06] = kernel_w
sm[07] = kernel_h
sm[08] = iy_stride
sm[09] = ix_stride
sm[10] = out_address
sm[11] = out_width
sm[12] = out_height
sm[13] = pooling_stride_w
sm[14] = pooling_stride_h
sm[15] = div_multiplier
sm[16] = div_shifter
sm[17] = sfr_size
sm[18] = sfr1
sm[19] = sfr2
sm[20] = sfr3
sm[21] = sfr4
sm[22] = sfr5
sm[23] = sfr6
sm[24] = sfr7
sm[25] = sfr8
