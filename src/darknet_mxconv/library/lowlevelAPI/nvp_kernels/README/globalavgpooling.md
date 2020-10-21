About
 - This kernel is for global avg pooling
 - div_multiplier and div_shifter is for division 1/{kernel_size} 
   e.g.) if you want 1/2 div_multiplier = 1, div_shifter = 1
   e.g.) if you want 1/3, approximate 1/3 to div_multiplier / 2^{div_shifter} form
 - It requires temporary space in LTC 
   * size of temporary space = iy_stride * 6
 - It requires cluster_size for each feature map
   e.g.) if one feature map requires 7 lane, cluster_size is 7
 - Output format
   * the only last lane in cluster has valid output
   * the otehr lanes have invalid output

NOTE
 - rounding is done by kernel

parameter list
sm[00] = in_address0
sm[01] = in_width0
sm[02] = in_address1
sm[03] = in_width1
sm[04] = in_height0
sm[05] = in_height1
sm[06] = iy_stride
sm[07] = ix_stride
sm[08] = cluster_size
sm[09] = out_address
sm[10] = temp_address
sm[11] = div_multiplier
sm[12] = div_shifter
sm[13] = sfr_size
sm[14] = sfr1
sm[15] = sfr2
sm[16] = sfr3
sm[17] = sfr4
sm[18] = sfr5
sm[19] = sfr6
sm[20] = sfr7
sm[21] = sfr8
