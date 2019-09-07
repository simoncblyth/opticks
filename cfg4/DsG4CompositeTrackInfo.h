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
// /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/G4DataHelpers/G4CompositeTrackInfo.h

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API DsG4CompositeTrackInfo : public G4VUserTrackInformation {
public:
     DsG4CompositeTrackInfo() : fHistoryTrackInfo(0), fPhotonTrackInfo(0) {} ;
     virtual ~DsG4CompositeTrackInfo();

     void SetHistoryTrackInfo(G4VUserTrackInformation* ti) { fHistoryTrackInfo=ti; }
     G4VUserTrackInformation* GetHistoryTrackInfo() { return fHistoryTrackInfo; }

     void SetPhotonTrackInfo(G4VUserTrackInformation* ti) { fPhotonTrackInfo=ti; }
     G4VUserTrackInformation* GetPhotonTrackInfo() { return fPhotonTrackInfo; }
     
     void Print() const {};
private:
     G4VUserTrackInformation* fHistoryTrackInfo;
     G4VUserTrackInformation* fPhotonTrackInfo;
};

#include "CFG4_TAIL.hh"



