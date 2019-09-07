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

#include "NTrianglesNPY.hpp"
#include "NPY.hpp"
#include "PLOG.hh"

void test_icosahedron_subdiv(unsigned int nsd)
{
    NTrianglesNPY* icos = NTrianglesNPY::icosahedron();

    NTrianglesNPY* tris = icos->subdivide(nsd);         
   
    unsigned int ntr = tris->getNumTriangles();

    LOG(info) << "test_icosahedron_subdiv" 
              << " nsd " << std::setw(4) << nsd
              << " ntr " << std::setw(6) << ntr 
              ; 
}


void test_icosahedron_subdiv()
{
    for(int i=0 ; i < 6 ; i++) test_icosahedron_subdiv(i) ;
}


void test_octahedron_subdiv(unsigned int nsd)
{
    NTrianglesNPY* oct = NTrianglesNPY::octahedron();

    NTrianglesNPY* tris = oct->subdivide(nsd);

    unsigned int ntr = tris->getNumTriangles();

    LOG(info) << "test_octahedron_subdiv" 
              << " nsd " << std::setw(4) << nsd
              << " ntr " << std::setw(6) << ntr 
              ;
}

void test_octahedron_subdiv()
{
    for(int i=0 ; i < 6 ; i++) test_octahedron_subdiv(i) ;
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_icosahedron_subdiv();
    test_octahedron_subdiv();

    return 0 ; 
}



