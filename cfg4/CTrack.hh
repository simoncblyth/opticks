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

// Simple G4Track wrapper providing getTrackStatusString()

class G4Track ;
#include "G4TrackStatus.hh"  
#include "CFG4_API_EXPORT.hh"
class CFG4_API CTrack {
   public:
    static const char* fAlive_ ;
    static const char* fStopButAlive_ ;
    static const char* fStopAndKill_ ;
    static const char* fKillTrackAndSecondaries_ ;
    static const char* fSuspend_ ;
    static const char* fPostponeToNextEvent_ ;
   public:
      static int Id(const G4Track* track);
      static int ParentId(const G4Track* track);
      static int StepId(const G4Track* track);
      static int PrimaryPhotonID(const G4Track* track);
      static float Wavelength(const G4Track* track);
      static float Wavelength(double thePhotonMomentum);
   public:
      CTrack(const G4Track* track);
      const char* getTrackStatusString();
      static const char* TrackStatusString(G4TrackStatus status);
   private:
      const G4Track* m_track ; 
};



