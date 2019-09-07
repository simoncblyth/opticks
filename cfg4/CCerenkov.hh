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

/**
CCerenkov
==========

All the specifics of setup and configuration of the 
Cerenkov process including choice of class implementation
are intended to be hidden inside the implementation of this class,
with only the base G4VProcess resulting from the configuration being 
exposed.

Possible values of --cerenkovclass option:

G4Cerenkov
   standard Geant4 implementation, from the current Geant4 version (at the moment 1042)

G4Cerenkov1042
   standard Geant4 implementation from 1042  

C4Cerenkov1042 *default*
   standard Geant4 implementation with addition of genstep collection, 
   the change can be controlled via compile definition WITH_OPTICKS_GENSTEP_COLLECTION  
   in CMakeLists.txt

**/

#include "CFG4_API_EXPORT.hh"

class CG4 ; 
class Opticks ; 
class G4VProcess ; 

class CFG4_API CCerenkov
{
    public:
        CCerenkov(const CG4* g4);
    private:
        G4VProcess* initProcess(const char* cls) const ;
    public:       
        G4VProcess* getProcess() const ; 
    private:
        const CG4*      m_g4 ; 
        const Opticks*  m_ok ; 
        const char*     m_class ; 
    private:
        G4int           m_maxNumPhotonsPerStep ;
        G4double        m_maxBetaChangePerStep ;  // maximum allowed change in beta = v/c in % (perCent)
        G4bool          m_trackSecondariesFirst ; // photons first before resuming with primaries
        G4int           m_verboseLevel ; 
    private:
        G4VProcess*     m_process ;  
};
 




