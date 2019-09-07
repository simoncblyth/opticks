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

template <typename T> class NPY ;
#include "plog/Severity.h"
#include "CFG4_API_EXPORT.hh"

/**
CPhotonCollector
===================

Canonical instance m_photon_collector resident of C4PhotonCollector



NB : **No Geant4 dependency** use C4PhotonCollector for that 

This only depends on NPY, so it can be relocated downwards 
to a future intermediary subproj above NPY but below G4 specifics.


Photons (item shape 4*4, 4 quads)
-------------------------------------

**/

class CFG4_API CPhotonCollector 
{
        static const plog::Severity LEVEL ; 
    public:
        static CPhotonCollector* Instance();
    public:
        CPhotonCollector();  

        std::string desc() const ;
        void Summary(const char* msg="CPhotonCollector::Summary") const  ;
    public:
        NPY<float>*  getPhoton() const ;
        void save(const char* path) const ; 
        void save(const char* dir, const char* name) const ; 
    public:
        void collectPhoton(
               double  pos_x,
               double  pos_y,
               double  pos_z,
               double  time,

               double  dir_x,
               double  dir_y,
               double  dir_z,
               double  weight,

               double  pol_x,
               double  pol_y,
               double  pol_z,
               double  wavelength,

               int flags_x,
               int flags_y,
               int flags_z,
               int flags_w
        );
    private:
        static CPhotonCollector* INSTANCE ;      
        NPY<float>*  m_photon ;
        unsigned     m_photon_itemsize ; 
        float*       m_photon_values ;  
        unsigned     m_photon_count ;  

};



