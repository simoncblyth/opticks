#pragma once

/**
sbibit.h : packs and unpacks 4*2bit integers into 8 bits  
-----------------------------------------------------------

**/

#define sbibit_PACK4( a, b, c, d)   ( \
       (( (a) & 0x3 ) <<  0 ) | \
       (( (b) & 0x3 ) <<  2 ) | \
       (( (c) & 0x3 ) <<  4 ) | \
       (( (d) & 0x3 ) <<  6 )   \
                              )

#define sbibit_UNPACK4_0( packed ) (  ((packed) >>  0) & 0x3 )
#define sbibit_UNPACK4_1( packed ) (  ((packed) >>  2) & 0x3 )
#define sbibit_UNPACK4_2( packed ) (  ((packed) >>  4) & 0x3 )
#define sbibit_UNPACK4_3( packed ) (  ((packed) >>  6) & 0x3 )


