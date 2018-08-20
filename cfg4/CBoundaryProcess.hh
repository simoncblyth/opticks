#pragma once

#include "CProcessSwitches.hh"

#ifdef USE_CUSTOM_BOUNDARY
#include "DsG4OpBoundaryProcess.h"
#else
#include "G4OpBoundaryProcess.hh"
#endif

#include "CFG4_API_EXPORT.hh"

class CFG4_API CBoundaryProcess
{
    public:
        // enu.py /usr/local/opticks/externals/g4/geant4.10.04.p02/source/processes/optical/include/G4OpBoundaryProcess.hh 
        static const char* Undefined_ ;
        static const char* Transmission_ ;
        static const char* FresnelRefraction_ ;
        static const char* FresnelReflection_ ;
        static const char* TotalInternalReflection_ ;
        static const char* LambertianReflection_ ;
        static const char* LobeReflection_ ;
        static const char* SpikeReflection_ ;
        static const char* BackScattering_ ;
        static const char* Absorption_ ;
        static const char* Detection_ ;
        static const char* NotAtBoundary_ ;
        static const char* SameMaterial_ ;
        static const char* StepTooSmall_ ;
        static const char* NoRINDEX_ ;
        static const char* PolishedLumirrorAirReflection_ ;
        static const char* PolishedLumirrorGlueReflection_ ;
        static const char* PolishedAirReflection_ ;
        static const char* PolishedTeflonAirReflection_ ;
        static const char* PolishedTiOAirReflection_ ;
        static const char* PolishedTyvekAirReflection_ ;
        static const char* PolishedVM2000AirReflection_ ;
        static const char* PolishedVM2000GlueReflection_ ;
        static const char* EtchedLumirrorAirReflection_ ;
        static const char* EtchedLumirrorGlueReflection_ ;
        static const char* EtchedAirReflection_ ;
        static const char* EtchedTeflonAirReflection_ ;
        static const char* EtchedTiOAirReflection_ ;
        static const char* EtchedTyvekAirReflection_ ;
        static const char* EtchedVM2000AirReflection_ ;
        static const char* EtchedVM2000GlueReflection_ ;
        static const char* GroundLumirrorAirReflection_ ;
        static const char* GroundLumirrorGlueReflection_ ;
        static const char* GroundAirReflection_ ;
        static const char* GroundTeflonAirReflection_ ;
        static const char* GroundTiOAirReflection_ ;
        static const char* GroundTyvekAirReflection_ ;
        static const char* GroundVM2000AirReflection_ ;
        static const char* GroundVM2000GlueReflection_ ;
        static const char* Dichroic_ ;
    public:
#ifdef USE_CUSTOM_BOUNDARY
        static Ds::DsG4OpBoundaryProcessStatus GetOpBoundaryProcessStatus() ;
        static const char* OpBoundaryString(const Ds::DsG4OpBoundaryProcessStatus status) ;
#else
        static G4OpBoundaryProcessStatus   GetOpBoundaryProcessStatus() ;
        static const char* OpBoundaryString(const G4OpBoundaryProcessStatus status) ;
#endif

};



