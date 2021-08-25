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
#include "plog/Severity.h"

template <typename T> class NPY ; 
struct NStep ; 
class TorchStepNPY ; 

class G4SPSPosDistribution ;
class G4SPSAngDistribution ;
class G4SPSEneDistribution ;
class G4SPSRandomGenerator ;

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"


/**
CTorchSource
=============

Canonical m_source instance lives in CGenerator
and is instanciated by CGenerator::initSource

**/


class CFG4_API CTorchSource: public CSource
{
        static const plog::Severity LEVEL ; 
    public:
        CTorchSource(Opticks* ok, TorchStepNPY* torch, unsigned int verbosity);
    private:
        void init();
        void configure();
    public:
        virtual ~CTorchSource();
        void setVerbosity(int verbosity) ;
        void GeneratePrimaryVertex(G4Event *evt);
        std::string desc() const ;
    private:
        TorchStepNPY*         m_torch ;
        NStep*                m_onestep ; 
        bool                  m_torchdbg ; 
        int                   m_verbosity ; 
        unsigned              m_num_photons_total ; 
        unsigned              m_num_photons_per_g4event ; 
        unsigned              m_num_photons ; 

        G4SPSPosDistribution* m_posGen;
        G4SPSAngDistribution* m_angGen;
        G4SPSEneDistribution* m_eneGen;
        G4SPSRandomGenerator* m_ranGen;

        NPY<float>*           m_primary ; 

};


