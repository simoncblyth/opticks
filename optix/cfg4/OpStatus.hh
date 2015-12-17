#pragma once

#include "G4StepStatus.hh"
#include "G4OpBoundaryProcess.hh"
#include <string>

std::string OpStepString(const G4StepStatus status);

std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status);
unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status);



