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
#ifdef USE_CUSTOM_BOUNDARY
        static DsG4OpBoundaryProcessStatus GetOpBoundaryProcessStatus() ;
#else
        static G4OpBoundaryProcessStatus   GetOpBoundaryProcessStatus() ;
#endif

};



