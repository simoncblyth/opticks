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
#include "G4Types.hh"

//class OpticksGenstep ; 
class NLookup ; 
template <typename T> class NPY ;

/**
CGenstepCollector : methods for collection of gensteps
=================================================================

Canonical CG4.m_collector is instanciated at postinitialize, 
and G4Opticks.m_collector instance is instanciated at setGeometry


G4 Independency 
----------------

NB there are no G4 classes used here, just a few type definitions
which could be removed easily. Users of this such as CSource should 
convert from G4 class instances into basic types for collection here.


Gensteps (item shape 6*4, 6 quads) 
-------------------------------------

First 3 quads are common between scintillation and
Cerenkov but the last 3 differ.

Furthermore the precise details of which quanties are included 
in the last three quads depends on the inner photon loop 
implementation details of the scintillation and Cerenkov processes.
They need to be setup whilst developing a corresponding GPU/OptiX 
implementation of the inner photon loop and doing 
equivalence comparisons.

Each implementation will need slightly different genstep and OptiX port.

Effectively the genstep can be regarded as the "stack" just 
prior to the photon generation loop.


 
**/

#include "CFG4_API_EXPORT.hh"

class CFG4_API CGenstepCollector 
{
   public:
         static CGenstepCollector* Instance();
   public:
         CGenstepCollector(const NLookup* lookup);  
   public:
         unsigned getNumGensteps() const ; 
         unsigned getNumPhotons() const ;  // total 
         unsigned getNumPhotons( unsigned gs_idx) const ; 
   public:
         NPY<float>*  getGensteps() const ;
   public:
         std::string description() const ;
         void Summary(const char* msg="CGenstepCollector::Summary") const  ;
         int translate(int acode) const ;
   private:
         //void setGensteps(NPY<float>* gs);
         void consistencyCheck() const ;
   public:
         void collectScintillationStep(
            G4int                id, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
            
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             meanVelocity, 

            G4int                scntId,
            G4double             slowerRatio,
            G4double             slowTimeConstant,
            G4double             slowerTimeConstant,

            G4double             scintillationTime,
            G4double             scintillationIntegrationMax,
            G4double             spare1=0,
            G4double             spare2=0
        );
   public:
         void collectCerenkovStep(
            G4int                id, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
            
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             preVelocity, 

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             postVelocity
        );
   public:
         void collectMachineryStep(unsigned code);
   private:
         static CGenstepCollector* INSTANCE ;      
   private:
         const NLookup*     m_lookup ; 

         NPY<float>*       m_genstep ;
         //OpticksGenstep*   m_gs ; 

         unsigned          m_genstep_itemsize ; 
         float*            m_genstep_values ;  
         unsigned          m_scintillation_count ; 
         unsigned          m_cerenkov_count ; 
         unsigned          m_machinery_count ; 

         std::vector<unsigned> m_gs_photons ; 

};
