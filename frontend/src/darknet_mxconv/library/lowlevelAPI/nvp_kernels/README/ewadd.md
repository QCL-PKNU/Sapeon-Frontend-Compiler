About
 - This kernel is for residual connection

NOTE)
 - by setting up approprimate scaler_bias, INT8 residual can be used in the form
 of UINT8 residual + relu 
 - residual output = (alpha * I0 + beta * I1 + scaler_bias) >> shifter

parameter list

sm[00] =  in_address0(first input)
sm[01] =  in_address1(second input)
sm[02] =  out_address
sm[03] =  in_width
sm[04] =  in_height
sm[05] =  dy_astride
sm[06] =  alpha(sclaer for first input)
sm[07] =  beta(scaler for second input)
sm[08] =  sclaer_bias
sm[09] =  shifter
