#pragma once

#include "X4_API_EXPORT.hh"

#include <string>
#include "G4ThreeVector.hh"

struct X4_API X4ThreeVector
{  
    static std::string Code( const G4ThreeVector& v, const char* identifier );
};


