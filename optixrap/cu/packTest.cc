// clang++ packTest.cc -lc++ -o $TMP/packTest && $TMP/packTest

#include "pack.h"
#include <cassert>
#include <iostream>

int main()
{
    unsigned a = 0xaa ; 
    unsigned b = 0xbb ; 
    unsigned c = 0xcc ; 
    unsigned d = 0xdd ; 

    assert( sizeof(unsigned) == 4 );

    unsigned packed = PACK4( a, b, c, d ) ;

    std::cout << " packed  " << std::hex << packed << std::endl ; 

    unsigned a1 = UNPACK4_0( packed );
    unsigned b1 = UNPACK4_1( packed );
    unsigned c1 = UNPACK4_2( packed );
    unsigned d1 = UNPACK4_3( packed );

    unsigned packed2 = PACK4( a1, b1, c1, d1 );

    std::cout << " packed2 " << std::hex << packed2 << std::endl ; 

    assert( a == a1 );
    assert( b == b1 );
    assert( c == c1 );
    assert( d == d1 );

    assert( packed2 == packed );

    unsigned fabricated = 0xddccbbaa ;
    unsigned a2 = UNPACK4_0( fabricated  ) ;
    unsigned b2 = UNPACK4_1( fabricated  ) ;
    unsigned c2 = UNPACK4_2( fabricated  ) ;
    unsigned d2 = UNPACK4_3( fabricated  ) ;

    
    assert( a2 == a );
    assert( b2 == b );
    assert( c2 == c );
    assert( d2 == d );




    return 0 ; 
}
