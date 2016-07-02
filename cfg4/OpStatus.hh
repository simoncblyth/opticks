#pragma once

#include "CFG4_PUSH.hh"
#include "G4StepStatus.hh"
#include "G4OpBoundaryProcess.hh"
#include "CFG4_POP.hh"

#include "CFG4_API_EXPORT.hh"

class G4StepPoint ; 

#include <string>

CFG4_API std::string OpStepString(const G4StepStatus status);

CFG4_API std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
CFG4_API std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status);
CFG4_API unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status);

CFG4_API unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst);


