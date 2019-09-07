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

// TEST=NPriTest om-t

#include "NPY.hpp"
#include "NPri.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    const char* path = "/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1/primaries.npy" ; 

    NPY<float>* p = NPY<float>::load(path) ; 
    if(!p) return 0 ; 

    NPri* pr = new NPri(p); 
    LOG(info) << std::endl << pr->desc(0); 

    int pdgcode = pr->getPDGCode(0) ; 
    LOG(info) << " pdgcode " << pdgcode ; 


    return 0 ; 
}
