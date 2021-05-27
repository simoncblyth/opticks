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

struct STranche ; 
class NPho ; 
template <typename T> class NPY ; 
class GenstepNPY ;



class G4PrimaryVertex ;

#include <string>
#include "plog/Severity.h"
#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

/**
CInputPhotonSource
====================

Canonical m_source instance lives in CGenerator, created by CGenerator::initInputPhotonSource

Implements the G4VPrimaryGenerator interface : GeneratePrimaryVertex



**/


class CFG4_API CInputPhotonSource: public CSource
{
        static const plog::Severity LEVEL ; 
    public:
        CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, unsigned numPhotonsPerG4Event );
        void reset(); 
    public:
        virtual ~CInputPhotonSource();
    public:
        // G4VPrimaryGenerator interface
        void     GeneratePrimaryVertex(G4Event *evt);
    public:
        unsigned  getNumG4Event() const ; 
        unsigned  getNumPhotonsPerG4Event() const;
        std::string desc() const ;
    private:
        G4PrimaryVertex*      convertPhoton(unsigned pho_index);
    private:
        bool                  m_sourcedbg ; 
        NPho*                 m_pho ;
    private:
        unsigned              m_numPhotonsPerG4Event ;
        unsigned              m_numPhotons ;
        STranche*             m_tranche ; 
        unsigned              m_gpv_count ;   // count calls to GeneratePrimaryVertex

};


