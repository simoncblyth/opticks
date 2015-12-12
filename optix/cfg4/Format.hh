#pragma once

#include "G4ThreeVector.hh"
#include "G4StepStatus.hh"
#include "G4OpBoundaryProcess.hh"

#include <string>

class G4StepPoint ; 
class G4Track ; 

std::string Format(const G4ThreeVector& vec, const char* msg="Vec");
std::string Format(const G4StepPoint* sp, const char* msg="StepPoint");
std::string Format(const G4Track* track, const char* msg="Track");
std::string Format(const G4Step* step, const char* msg="Step");

std::string Format(const G4StepStatus status);
std::string Format(const G4OpBoundaryProcessStatus status);
 
