#pragma once

#include <string>
#include "G4ThreeVector.hh"

struct CThreeVector
{
    static std::string Format(const G4ThreeVector& vec, int width=10, int precision=3 ); 
};

