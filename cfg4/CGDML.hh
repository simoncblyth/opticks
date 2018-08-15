#pragma once

#include "CFG4_API_EXPORT.hh"

/**
**/

class G4VPhysicalVolume ; 

class CFG4_API CGDML
{
    public:
        static void Export(const char* dir, const char* name, const G4VPhysicalVolume* const world );
        static void Export(const char* path, const G4VPhysicalVolume* const world );
};


 
