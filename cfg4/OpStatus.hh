#pragma once

#include "CFG4_PUSH.hh"
#include "G4StepStatus.hh"
#include "CBoundaryProcess.hh"
#include "CFG4_POP.hh"

#include "CFG4_API_EXPORT.hh"

class G4StepPoint ; 

#include <string>

CFG4_API std::string OpStepString(const G4StepStatus status);


#ifdef USE_CUSTOM_BOUNDARY
CFG4_API std::string  OpBoundaryString(const DsG4OpBoundaryProcessStatus status);
CFG4_API std::string OpBoundaryAbbrevString(const DsG4OpBoundaryProcessStatus status);
CFG4_API unsigned int OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status);
CFG4_API unsigned int OpPointFlag(const G4StepPoint* point, const DsG4OpBoundaryProcessStatus bst);
#else
CFG4_API std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
CFG4_API std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status);
CFG4_API unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status);
CFG4_API unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst);
#endif


