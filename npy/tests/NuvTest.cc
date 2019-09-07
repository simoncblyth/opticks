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

#include "Nuv.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    unsigned s = 0 ; 
    unsigned u = 1 ; 
    unsigned v = 2 ; 
    unsigned nu = 10 ; 
    unsigned nv = 20 ; 
    unsigned pr = 0 ; 

    nuv p = make_uv(s,u,v,nu,nv, pr);

    assert( p.s() == s ) ; 
    assert( p.u() == u ) ; 
    assert( p.v() == v ) ; 
    assert( p.nu() == nu ); 
    assert( p.nv() == nv ); 
    assert( p.nv() == nv ); 
    assert( p.p() == pr ); 

    LOG(info) << p.desc() ; 


    return 0 ; 
}
