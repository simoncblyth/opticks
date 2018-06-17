#pragma once

#include "X4_API_EXPORT.hh"
#include <string>

class G4Material ; 
class G4LogicalSurface ; 

class X4_API X4 
{
    public: 
        static const char* ShortName( const G4Material* const material );
        static const char* ShortName( const G4LogicalSurface* const surface );
        static const char* ShortName( const std::string& name );
};


