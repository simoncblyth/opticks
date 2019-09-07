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

#pragma once


#include <glm/fwd.hpp>

template <typename T> class NPY ;
class NTrianglesNPY ; 

//
//  *NTesselate*  just does triangle subdivision  
//         
//  Delaunay Tessellation is the general approach to this 
//  for a comparison of available code:
//
//  * http://library.msri.org/books/Book52/files/23liu.pdf
//
//
struct ntriangle ; 
#include "NPY_API_EXPORT.hh"

class NPY_API NTesselate {
    public:
        NTesselate(NPY<float>* basis);
        void subdivide(unsigned int nsubdiv);
        void add(glm::vec3& a, glm::vec3& c, const glm::vec3& v);
        NPY<float>* getBuffer();
    private:
        void init(); 
        void subdivide(unsigned int nsubdiv, ntriangle& t);
    private:
        NPY<float>*    m_basis ; 
        NTrianglesNPY* m_tris ; 
};



