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

#include "G4ProcessManager.hh"
#include "CBoundaryProcess.hh"
#include "PLOG.hh"


#ifdef USE_CUSTOM_BOUNDARY
Ds::DsG4OpBoundaryProcessStatus CBoundaryProcess::GetOpBoundaryProcessStatus()
{
    Ds::DsG4OpBoundaryProcessStatus status = Ds::Undefined;
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    assert( mgr );
    if(mgr) 
    {
        DsG4OpBoundaryProcess* opProc = NULL ;  
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];
            opProc = dynamic_cast<DsG4OpBoundaryProcess*>(proc);
            if (opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
            // NULL casts expected, that is the way of finding the wanted boundary process
        }
    }
    return status ; 
}
#else
G4OpBoundaryProcessStatus CBoundaryProcess::GetOpBoundaryProcessStatus()
{
    G4OpBoundaryProcessStatus status = Undefined;
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    if(mgr) 
    {
        G4OpBoundaryProcess* opProc = NULL ;  
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];
            opProc = dynamic_cast<G4OpBoundaryProcess*>(proc);
            if (opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
            // NULL casts expected, that is the way of finding the wanted boundary process
        }
    }
    return status ; 
}
#endif


// enu.py /usr/local/opticks/externals/g4/geant4.10.04.p02/source/processes/optical/include/G4OpBoundaryProcess.hh 
const char* CBoundaryProcess::Undefined_                          = "Undefined"                         ;
const char* CBoundaryProcess::Transmission_                       = "Transmission"                      ;
const char* CBoundaryProcess::FresnelRefraction_                  = "FresnelRefraction"                 ;
const char* CBoundaryProcess::FresnelReflection_                  = "FresnelReflection"                 ;
const char* CBoundaryProcess::TotalInternalReflection_            = "TotalInternalReflection"           ;
const char* CBoundaryProcess::LambertianReflection_               = "LambertianReflection"              ;
const char* CBoundaryProcess::LobeReflection_                     = "LobeReflection"                    ;
const char* CBoundaryProcess::SpikeReflection_                    = "SpikeReflection"                   ;
const char* CBoundaryProcess::BackScattering_                     = "BackScattering"                    ;
const char* CBoundaryProcess::Absorption_                         = "Absorption"                        ;
const char* CBoundaryProcess::Detection_                          = "Detection"                         ;
const char* CBoundaryProcess::NotAtBoundary_                      = "NotAtBoundary"                     ;
const char* CBoundaryProcess::SameMaterial_                       = "SameMaterial"                      ;
const char* CBoundaryProcess::StepTooSmall_                       = "StepTooSmall"                      ;
const char* CBoundaryProcess::NoRINDEX_                           = "NoRINDEX"                          ;
const char* CBoundaryProcess::PolishedLumirrorAirReflection_      = "PolishedLumirrorAirReflection"     ;
const char* CBoundaryProcess::PolishedLumirrorGlueReflection_     = "PolishedLumirrorGlueReflection"    ;
const char* CBoundaryProcess::PolishedAirReflection_              = "PolishedAirReflection"             ;
const char* CBoundaryProcess::PolishedTeflonAirReflection_        = "PolishedTeflonAirReflection"       ;
const char* CBoundaryProcess::PolishedTiOAirReflection_           = "PolishedTiOAirReflection"          ;
const char* CBoundaryProcess::PolishedTyvekAirReflection_         = "PolishedTyvekAirReflection"        ;
const char* CBoundaryProcess::PolishedVM2000AirReflection_        = "PolishedVM2000AirReflection"       ;
const char* CBoundaryProcess::PolishedVM2000GlueReflection_       = "PolishedVM2000GlueReflection"      ;
const char* CBoundaryProcess::EtchedLumirrorAirReflection_        = "EtchedLumirrorAirReflection"       ;
const char* CBoundaryProcess::EtchedLumirrorGlueReflection_       = "EtchedLumirrorGlueReflection"      ;
const char* CBoundaryProcess::EtchedAirReflection_                = "EtchedAirReflection"               ;
const char* CBoundaryProcess::EtchedTeflonAirReflection_          = "EtchedTeflonAirReflection"         ;
const char* CBoundaryProcess::EtchedTiOAirReflection_             = "EtchedTiOAirReflection"            ;
const char* CBoundaryProcess::EtchedTyvekAirReflection_           = "EtchedTyvekAirReflection"          ;
const char* CBoundaryProcess::EtchedVM2000AirReflection_          = "EtchedVM2000AirReflection"         ;
const char* CBoundaryProcess::EtchedVM2000GlueReflection_         = "EtchedVM2000GlueReflection"        ;
const char* CBoundaryProcess::GroundLumirrorAirReflection_        = "GroundLumirrorAirReflection"       ;
const char* CBoundaryProcess::GroundLumirrorGlueReflection_       = "GroundLumirrorGlueReflection"      ;
const char* CBoundaryProcess::GroundAirReflection_                = "GroundAirReflection"               ;
const char* CBoundaryProcess::GroundTeflonAirReflection_          = "GroundTeflonAirReflection"         ;
const char* CBoundaryProcess::GroundTiOAirReflection_             = "GroundTiOAirReflection"            ;
const char* CBoundaryProcess::GroundTyvekAirReflection_           = "GroundTyvekAirReflection"          ;
const char* CBoundaryProcess::GroundVM2000AirReflection_          = "GroundVM2000AirReflection"         ;
const char* CBoundaryProcess::GroundVM2000GlueReflection_         = "GroundVM2000GlueReflection"        ;
const char* CBoundaryProcess::Dichroic_                           = "Dichroic"                          ;



#ifdef USE_CUSTOM_BOUNDARY
const char* CBoundaryProcess::OpBoundaryString(const Ds::DsG4OpBoundaryProcessStatus status)
{
    const char* s = NULL ; 
    switch(status)
    {
        // enu.py DsG4OpBoundaryProcessStatus.h 
        case Ds::Undefined                 : s = Undefined_                ; break ;
        case Ds::FresnelRefraction         : s = FresnelRefraction_        ; break ;
        case Ds::FresnelReflection         : s = FresnelReflection_        ; break ;
        case Ds::TotalInternalReflection   : s = TotalInternalReflection_  ; break ;
        case Ds::LambertianReflection      : s = LambertianReflection_     ; break ;
        case Ds::LobeReflection            : s = LobeReflection_           ; break ;
        case Ds::SpikeReflection           : s = SpikeReflection_          ; break ;
        case Ds::BackScattering            : s = BackScattering_           ; break ;
        case Ds::Absorption                : s = Absorption_               ; break ;
        case Ds::Detection                 : s = Detection_                ; break ;
        case Ds::NotAtBoundary             : s = NotAtBoundary_            ; break ;
        case Ds::SameMaterial              : s = SameMaterial_             ; break ;
        case Ds::StepTooSmall              : s = StepTooSmall_             ; break ;
        case Ds::NoRINDEX                  : s = NoRINDEX_                 ; break ;
    }
    return s ;
}
#else
const char* CBoundaryProcess::OpBoundaryString(const G4OpBoundaryProcessStatus status)
{
    const char* s = NULL ; 
    switch(status)
    {
        // enu.py /usr/local/opticks/externals/g4/geant4.10.04.p02/source/processes/optical/include/G4OpBoundaryProcess.hh 
        case Undefined                           : s = Undefined_                          ; break ;
        case Transmission                        : s = Transmission_                       ; break ;
        case FresnelRefraction                   : s = FresnelRefraction_                  ; break ;
        case FresnelReflection                   : s = FresnelReflection_                  ; break ;
        case TotalInternalReflection             : s = TotalInternalReflection_            ; break ;
        case LambertianReflection                : s = LambertianReflection_               ; break ;
        case LobeReflection                      : s = LobeReflection_                     ; break ;
        case SpikeReflection                     : s = SpikeReflection_                    ; break ;
        case BackScattering                      : s = BackScattering_                     ; break ;
        case Absorption                          : s = Absorption_                         ; break ;
        case Detection                           : s = Detection_                          ; break ;
        case NotAtBoundary                       : s = NotAtBoundary_                      ; break ;
        case SameMaterial                        : s = SameMaterial_                       ; break ;
        case StepTooSmall                        : s = StepTooSmall_                       ; break ;
        case NoRINDEX                            : s = NoRINDEX_                           ; break ;
        case PolishedLumirrorAirReflection       : s = PolishedLumirrorAirReflection_      ; break ;
        case PolishedLumirrorGlueReflection      : s = PolishedLumirrorGlueReflection_     ; break ;
        case PolishedAirReflection               : s = PolishedAirReflection_              ; break ;
        case PolishedTeflonAirReflection         : s = PolishedTeflonAirReflection_        ; break ;
        case PolishedTiOAirReflection            : s = PolishedTiOAirReflection_           ; break ;
        case PolishedTyvekAirReflection          : s = PolishedTyvekAirReflection_         ; break ;
        case PolishedVM2000AirReflection         : s = PolishedVM2000AirReflection_        ; break ;
        case PolishedVM2000GlueReflection        : s = PolishedVM2000GlueReflection_       ; break ;
        case EtchedLumirrorAirReflection         : s = EtchedLumirrorAirReflection_        ; break ;
        case EtchedLumirrorGlueReflection        : s = EtchedLumirrorGlueReflection_       ; break ;
        case EtchedAirReflection                 : s = EtchedAirReflection_                ; break ;
        case EtchedTeflonAirReflection           : s = EtchedTeflonAirReflection_          ; break ;
        case EtchedTiOAirReflection              : s = EtchedTiOAirReflection_             ; break ;
        case EtchedTyvekAirReflection            : s = EtchedTyvekAirReflection_           ; break ;
        case EtchedVM2000AirReflection           : s = EtchedVM2000AirReflection_          ; break ;
        case EtchedVM2000GlueReflection          : s = EtchedVM2000GlueReflection_         ; break ;
        case GroundLumirrorAirReflection         : s = GroundLumirrorAirReflection_        ; break ;
        case GroundLumirrorGlueReflection        : s = GroundLumirrorGlueReflection_       ; break ;
        case GroundAirReflection                 : s = GroundAirReflection_                ; break ;
        case GroundTeflonAirReflection           : s = GroundTeflonAirReflection_          ; break ;
        case GroundTiOAirReflection              : s = GroundTiOAirReflection_             ; break ;
        case GroundTyvekAirReflection            : s = GroundTyvekAirReflection_           ; break ;
        case GroundVM2000AirReflection           : s = GroundVM2000AirReflection_          ; break ;
        case GroundVM2000GlueReflection          : s = GroundVM2000GlueReflection_         ; break ;
        case Dichroic                            : s = Dichroic_                           ; break ;
    }
    return s ;
}
#endif


