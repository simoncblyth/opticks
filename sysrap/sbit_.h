#pragma once

/**
sbit_.h : packs and unpacks 8*1bit bools into 8 bits  
-----------------------------------------------------------

The r variant flips the packing order

**/

#define sbit_PACK8( a, b, c, d, e, f, g, h)   ( \
       (( (a) & 0x1 ) <<  0 ) | \
       (( (b) & 0x1 ) <<  1 ) | \
       (( (c) & 0x1 ) <<  2 ) | \
       (( (d) & 0x1 ) <<  3 ) | \
       (( (e) & 0x1 ) <<  4 ) | \
       (( (f) & 0x1 ) <<  5 ) | \
       (( (g) & 0x1 ) <<  6 ) | \
       (( (h) & 0x1 ) <<  7 )   \
                              )

#define sbit_UNPACK8_0( packed ) (  ((packed) >>  0) & 0x1 )
#define sbit_UNPACK8_1( packed ) (  ((packed) >>  1) & 0x1 )
#define sbit_UNPACK8_2( packed ) (  ((packed) >>  2) & 0x1 )
#define sbit_UNPACK8_3( packed ) (  ((packed) >>  3) & 0x1 )
#define sbit_UNPACK8_4( packed ) (  ((packed) >>  4) & 0x1 )
#define sbit_UNPACK8_5( packed ) (  ((packed) >>  5) & 0x1 )
#define sbit_UNPACK8_6( packed ) (  ((packed) >>  6) & 0x1 )
#define sbit_UNPACK8_7( packed ) (  ((packed) >>  7) & 0x1 )




#define sbit_rPACK8( a, b, c, d, e, f, g, h)   ( \
       (( (a) & 0x1 ) <<  7 ) | \
       (( (b) & 0x1 ) <<  6 ) | \
       (( (c) & 0x1 ) <<  5 ) | \
       (( (d) & 0x1 ) <<  4 ) | \
       (( (e) & 0x1 ) <<  3 ) | \
       (( (f) & 0x1 ) <<  2 ) | \
       (( (g) & 0x1 ) <<  1 ) | \
       (( (h) & 0x1 ) <<  0 )   \
                              )

#define sbit_rUNPACK8_0( packed ) (  ((packed) >>  7) & 0x1 )
#define sbit_rUNPACK8_1( packed ) (  ((packed) >>  6) & 0x1 )
#define sbit_rUNPACK8_2( packed ) (  ((packed) >>  5) & 0x1 )
#define sbit_rUNPACK8_3( packed ) (  ((packed) >>  4) & 0x1 )
#define sbit_rUNPACK8_4( packed ) (  ((packed) >>  3) & 0x1 )
#define sbit_rUNPACK8_5( packed ) (  ((packed) >>  2) & 0x1 )
#define sbit_rUNPACK8_6( packed ) (  ((packed) >>  1) & 0x1 )
#define sbit_rUNPACK8_7( packed ) (  ((packed) >>  0) & 0x1 )


