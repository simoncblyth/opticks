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

#include <cassert>

#include "GBuffer.hh"
#include "GArray.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    LOG(info) << argv[0] ;

    float v[3] ;
    v[0] = 1.f ; 
    v[1] = 1.f ; 
    v[2] = 1.f ; 

    GArray<float>* a = new GArray<float>(3, v );
    assert( a->getLength() == 3 );


    const char* path = "$TMP/GArrayTest.npy" ;
    LOG(info) << "saving to " << path ; 

    a->save<float>(path);




    return 0 ;
}

