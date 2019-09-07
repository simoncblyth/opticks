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

class Opticks ; 

template <typename T> class NPY ;
#include "G4Types.hh"
#include "CFG4_API_EXPORT.hh"
#include "CSource.hh"

/**
CPrimaryCollector
===================

Primaries (item shape 4*4, 4 quads)
-------------------------------------

Primary collection is invoked from CSource::collectPrimary(G4PrimaryVertex* vtx)
into the CPrimaryCollector singleton instance.

**/

class CFG4_API CPrimaryCollector 
{
    public:
        static CPrimaryCollector* Instance();
    private:
        static CPrimaryCollector* INSTANCE ;      
    public:
        CPrimaryCollector();  
        std::string description() const ;
        void Summary(const char* msg="CPrimaryCollector::Summary") const  ;
    public:
        NPY<float>*  getPrimary() const ;
        void save(const char* path) const ; 

    public:
       //void collectPrimary(G4PrimaryVertex* vertex);
       void collectPrimaries(const G4Event* event);
       void collectPrimaryVertex(G4int vertex_index, const G4Event* event);
       void collectPrimaryVertex(const G4PrimaryVertex* vtx, G4int vertex_index);   // for backward compat to CSource collection
       void collectPrimaryParticle(G4int vertex_index, G4int primary_index, const G4PrimaryVertex* vtx);

        void collectPrimary(
               G4double  x0,
               G4double  y0,
               G4double  z0,
               G4double  t0,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  kineticEnergy,

               int spare,
               int vertex_index,
               int primary_index,
               int pdgcode
        );
    private:
        NPY<float>*  m_primary ;
        unsigned     m_primary_itemsize ; 
        float*       m_primary_values ;  
        unsigned     m_primary_count ;  

};



