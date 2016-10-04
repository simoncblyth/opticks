#pragma once

#include "CProcessSwitches.hh"

#ifdef USE_CUSTOM_BOUNDARY
#include "DsG4OpBoundaryProcess.h"
#else
#include "G4OpBoundaryProcess.hh"
#endif

