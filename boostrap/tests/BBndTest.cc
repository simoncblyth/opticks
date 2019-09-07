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


#include <cstring>
#include "BBnd.hh"
#include "PLOG.hh"



void test_DuplicateOuterMaterial()
{
    const char* b0 = "Rock///Pyrex" ; 

    const char* b1 = BBnd::DuplicateOuterMaterial(b0) ;
    const char* x1 = "Rock///Rock" ; 

    if( strcmp(b1,x1) != 0 )
    {
        LOG(error) << " b1 [" << b1 << "]"
                   << " x1 [" << x1 << "]"
                   ;
    }
    assert( strcmp(b1,x1) == 0 ); 
}


void test_BBnd()
{
    {
        BBnd b("omat/osur/isur/imat");
        LOG(info) << b.desc() ; 
        assert( b.omat && b.osur && b.isur && b.imat );  
    }

    {
        BBnd b("omat///imat");
        LOG(info) << b.desc() ; 
        assert( b.omat && !b.osur && !b.isur && b.imat );  
    }
   
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_DuplicateOuterMaterial();
    test_BBnd();


    return 0 ; 
}
 
