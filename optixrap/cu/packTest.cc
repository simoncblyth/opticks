/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
