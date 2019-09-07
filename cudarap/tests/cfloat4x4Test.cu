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

#include "OPTICKS_LOG.hh"
#include "cfloat4x4.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    cfloat4x4 pho ; 

    pho.q0 = make_float4( 0.5f,   1.5f,  2.5f,  3.5f );     
    pho.q1 = make_float4( 10.5f, 11.5f, 12.5f, 13.5f );     
    pho.q2 = make_float4( 20.5f, 21.5f, 22.5f, 23.5f );     

    tquad q3 ; 
    q3.u = make_uint4(    42, 43, 44, 45 );  
    pho.q3 = q3.f ; 

    LOG(info) << pho ; 

    return 0 ; 
}


