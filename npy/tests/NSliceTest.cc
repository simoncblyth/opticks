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

#include "NSlice.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>


void test_slice(const char* arg)
{
    NSlice* s = new NSlice(arg) ;
    printf("arg %s slice %s \n", arg, s->description()) ; 
}

void test_margin()
{
    NSlice s(0,10,1);

    assert(  s.isHead(0,2) );
    assert(  s.isHead(1,2) );
    assert( !s.isHead(2,2) );
    assert( !s.isHead(3,2) );
    assert( !s.isHead(4,2) );
    assert( !s.isHead(5,2) );
    assert( !s.isHead(6,2) );
    assert( !s.isHead(7,2) );
    assert( !s.isHead(8,2) );
    assert( !s.isHead(9,2) );
    assert( !s.isHead(10,2) );

    assert( !s.isTail(0,2) );
    assert( !s.isTail(1,2) );
    assert( !s.isTail(2,2) );
    assert( !s.isTail(3,2) );
    assert( !s.isTail(4,2) );
    assert( !s.isTail(5,2) );
    assert( !s.isTail(6,2) );
    assert( !s.isTail(7,2) );
    assert(  s.isTail(8,2) );
    assert(  s.isTail(9,2) );
    assert( !s.isTail(10,2) );

    assert(  s.isMargin(0,2) );
    assert(  s.isMargin(1,2) );
    assert( !s.isMargin(2,2) );
    assert( !s.isMargin(3,2) );
    assert( !s.isMargin(4,2) );
    assert( !s.isMargin(5,2) );
    assert( !s.isMargin(6,2) );
    assert( !s.isMargin(7,2) );
    assert(  s.isMargin(8,2) );
    assert(  s.isMargin(9,2) );
    assert( !s.isMargin(10,2) );
}



int main()
{
    test_slice("0:10");
    test_slice("0:10:2");

    test_margin();

    return 0 ;
}
