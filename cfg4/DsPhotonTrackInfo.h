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

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API DsPhotonTrackInfo : public G4VUserTrackInformation
{
public:
    enum QEMode 
    { 
            kQENone, 
            kQEPreScale, 
            kQEWater 
    };

    DsPhotonTrackInfo(QEMode mode=DsPhotonTrackInfo::kQENone, double qe=1.) ;


    QEMode GetMode() { return fMode; }
    void   SetMode(QEMode m) { fMode=m; }

    double GetQE() { return fQE; }
    void   SetQE(double qe) { fQE=qe; }

    bool GetReemitted() { return fReemitted; }
    void SetReemitted( bool re=true ) { fReemitted=re; }

    void  SetPrimaryPhotonID(int ppi){ fPrimaryPhotonID = ppi ; ; }
    int   GetPrimaryPhotonID(){ return fPrimaryPhotonID ; } 
    
    void Print() const {};
private:
    QEMode fMode;
    double fQE;
    bool   fReemitted;
    int    fPrimaryPhotonID  ; 
};

#include "CFG4_TAIL.hh"
