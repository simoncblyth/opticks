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

#include <string>

#include "NBBox.hpp"
#include "NOpenMeshType.hpp"

#include "NPY_API_EXPORT.hh"

template <typename T> struct NOpenMesh ; 

#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ;
#endif

class NTrianglesNPY ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NHybridMesher
{
    public:
         typedef NOpenMesh<NOpenMeshType> MESH ; 
     public:
#ifdef OLD_PARAMETERS
        NHybridMesher(const nnode* node, X_BParameters* meta, const char* treedir=NULL );
#else
        NHybridMesher(const nnode* node, NMeta* meta, const char* treedir=NULL );
#endif
        NTrianglesNPY* operator()();
        std::string desc();
    private:
        void init(); 
    private:
        NOpenMesh<NOpenMeshType>*  m_mesh ;
        nbbox*                     m_bbox ; 
        const char*                m_treedir ; 
  

};
