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

class OpticksHub ; 
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ; 
class CG4 ; 
class TorchStepNPY ; 
class CSource ; 
template <typename T> class NPY ; 

#include "plog/Severity.h"
#include "CFG4_API_EXPORT.hh"

/**

CGenerator
===========

Canonical m_generator instance is CG4 resident instanciated within it.    

Instanciation hooks up the configured sources (of photons or primaries) 
into CSource(G4VPrimaryGenerator) ready to be given to Geant4
to make primary vertices with. 

=====================  ==========
source                  dynamic 
=====================  ==========
inputPhotonSource         N 
TorchSource               N
G4GunSource               Y
=====================  ==========


Used from CG4::CG4 initializer list to prepare the CSource instance
to pass to CPrimaryGeneratorAction::

    123     m_generator(new CGenerator(m_hub->getGen(), this)),
    124     m_dynamic(m_generator->isDynamic()),
    ...
    132     m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),


**/

class CFG4_API CGenerator 
{
       static const plog::Severity LEVEL ; 
   public:
       CGenerator(OpticksGen* gen, CG4* g4);
   public:
       void        configureEvent(OpticksEvent* evt);
   public:
       unsigned    getSourceCode() const ;
       const char* getSourceType() const ;
       CSource*    getSource() const ;
       bool        isDynamic() const ;
       unsigned    getNumG4Event() const ;
       unsigned    getNumPhotonsPerG4Event() const ;
       NPY<float>* getGensteps() const ;
       bool        hasGensteps() const ;
       NPY<float>* getSourcePhotons() const ; // prior to propagation
   private:
       void init();
       CSource* initSource(unsigned code);
       CSource* initInputPhotonSource();
       CSource* initInputGenstepSource();
       CSource* initTorchSource();
       CSource* initG4GunSource();
    private:
       void setDynamic(bool dynamic);
       void setNumG4Event(unsigned num);
       void setNumPhotonsPerG4Event(unsigned num);
       void setNumGenstepsPerG4Event(unsigned num);
       void setGensteps(NPY<float>* gensteps);
   private:
       OpticksGen*           m_gen ;
       Opticks*              m_ok ;
       OpticksCfg<Opticks>*  m_cfg ;
       CG4*                  m_g4 ; 
   private:
       unsigned              m_source_code ; 
       const char*           m_source_type ; 
       NPY<float>*           m_gensteps ; 
       bool                  m_dynamic ; 
       unsigned              m_num_g4evt ; 
       unsigned              m_photons_per_g4evt ;           
       unsigned              m_gensteps_per_g4evt ;
   private:
       CSource*              m_source ; 

};


