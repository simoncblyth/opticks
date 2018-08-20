#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"

/**
**/

class G4VPhysicalVolume ; 

class CFG4_API CGDML
{
    public:
        static void Export(const char* dir, const char* name, const G4VPhysicalVolume* const world );
        static void Export(const char* path, const G4VPhysicalVolume* const world );
        static std::string GenerateName(const char* name, const void* const ptr, bool addPointerToName=true );

};


 
