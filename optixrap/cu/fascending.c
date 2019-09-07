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

// clang fascending.c && ./a.out && rm a.out

#include "assert.h"
#include "fascending.h"

void test_fascending_ptr()
{
    float a[4] ; 

    a[0] = 3 ; 
    a[1] = 2 ; 
    a[2] = 1 ; 
    a[3] = 0 ; 

    fascending_ptr(3, a );

    assert( a[0] == 1 );
    assert( a[1] == 2 );
    assert( a[2] == 3 );


    a[0] = 10 ; 
    a[1] = 5 ; 

    fascending_ptr(2, a );

    assert( a[0] == 5 );
    assert( a[1] == 10 );
     

    a[0] = 101 ; 
    fascending_ptr(1, a );
    assert( a[0] == 101 );


    a[0] = 300 ; 
    a[1] = 200 ; 
    a[2] = 700 ; 
    a[3] = 600 ; 

    fascending_ptr(4, a );

    assert( a[0] == 200 );
    assert( a[1] == 300 );
    assert( a[2] == 600 );
    assert( a[3] == 700 );
}


int main()
{
    test_fascending_ptr() ;
    return 0 ; 
}
