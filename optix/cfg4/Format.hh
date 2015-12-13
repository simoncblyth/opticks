#pragma once

#include <string>

class G4Track ; 
class G4Step ; 
class G4StepPoint ; 
#include "G4ThreeVector.hh"

std::string Format(const G4Track* track, const char* msg="Track");
std::string Format(const G4Step* step, const char* msg="Step");
std::string Format(const G4StepPoint* sp, const char* msg="StepPoint");
std::string Format(const G4ThreeVector& vec, const char* msg="Vec", unsigned int fwid=4);




 
