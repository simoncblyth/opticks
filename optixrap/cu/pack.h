#pragma once

#define PACK4( a, b, c, d)   ( \
       (( (a) & 0xff ) <<  0 ) | \
       (( (b) & 0xff ) <<  8 ) | \
       (( (c) & 0xff ) << 16 ) | \
       (( (d) & 0xff ) << 24 ) \
                             )

#define UNPACK4_0( packed ) (  ((packed) >>  0) & 0xff )
#define UNPACK4_1( packed ) (  ((packed) >>  8) & 0xff )
#define UNPACK4_2( packed ) (  ((packed) >> 16) & 0xff )
#define UNPACK4_3( packed ) (  ((packed) >> 24) & 0xff )


