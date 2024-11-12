About
 - This kernel is for depthwise convolution whose kernel size is not defined
 - filter packing = int32 * 2 for each c_word * filter_w * filter_w 
   * word requires 64bit (in NCHW format)
   * filter[0:31] = filter_lowerbit, filter[32:63] = filter_higherbit
   * filter_lower_bit -> lsb -> lower channel, msb -> higher channel
 - bias packing = int32 * 6 for each c_word (word requires 192bit for acc)
   * (In case of INT16) bias[0:47] = channel #0 bias[48:95] = channel #1 ...
   * (In case of INT8) bias[0:23] = channel #0 bias[24:47] = channel #1 ...

NOTE
 - bias shuold have rounding value (1 << (shift - 1))
 - because there are two shifts( (acc >> s0) * m >> s1 ), two bias shuold be
 offered
   * bias0: convolution bias + (1 << (s0 - 1))
   * bias1: 1 << (s1 - 1)


parameter list

sm[00] =  in_address0
sm[01] =  in_width0
sm[02] =  in_address1
sm[03] =  in_width1
sm[04] =  in_height0
sm[05] =  in_height1
sm[06] =  filter_w
sm[07] =  filter_h
sm[08] =  iy_stride
sm[09] =  ix_stride
sm[10] =  out_address
sm[11] =  out_width
sm[12] =  out_height
sm[13] =  pooling_stride_w
sm[14] =  pooling_stride_h
sm[15] =  num_cwords // cwords to be processed
sm[16] =  pad_top
sm[17] =  pad_left
sm[18] =  sfr_size
sm[19 ~ 26] = sfrs
sm[27] =  filter_size
sm[28] =  bias_size // bias size for each bias0 and bias1
sm[29] =  mult_size
sm[30] =  shf_size // shifte size for each sht0 and sht1
sm[31 ~ filter_size - 1] = filters
sm[31 + filter_size ~ bias_size - 1] = bias0
sm[31 + filter_size + bias_size + bias_size - 1] = bias1
sm[..] = multiplier
sm[..] = shfs0
sm[..] = shfs1
