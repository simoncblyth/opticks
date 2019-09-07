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

#include "X4_API_EXPORT.hh"

class G4VSolid ; 
class G4Polyhedron ; 

template <typename T> class NPY ;
class GMesh ; 

/**
X4Mesh
========

Uses polygonization from G4Polyhedron to convert 
a G4VSolid into a GMesh.

**/

class X4_API X4Mesh
{
    public:
        static GMesh* Placeholder(const G4VSolid* solid );
        static GMesh* Convert(const G4VSolid* solid );
    public:
        X4Mesh(const G4VSolid* solid); 
        std::string desc() const  ; 
        void save(const char* path="/tmp/X4Mesh/name.gltf") const  ; 
        GMesh* getMesh() const ;
    private:
        void init();
    private:
        void polygonize();
        void collect();
        void makemesh();
    private:
        void collect_vtx(int ivert);
        void collect_raw(int iface);
        void collect_tri();
    private:
        const G4VSolid* m_solid ;  
        G4Polyhedron*   m_polyhedron ;
        NPY<float>*     m_vtx ; 
        NPY<unsigned>*  m_raw ; // tris or quads
        NPY<unsigned>*  m_tri ; // tris by splitting quads  
        GMesh*          m_mesh ; 
        int             m_verbosity ; 

};

