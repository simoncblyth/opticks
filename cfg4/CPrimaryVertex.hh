#pragma once

#include <string>
class G4PrimaryVertex ; 

struct CPrimaryVertex 
{
    static const G4int OPTICAL_PHOTON_CODE ; 
    static bool IsInputPhoton(const G4PrimaryVertex* vtx ); 
    static std::string Desc(const G4PrimaryVertex* vtx ); 
}; 
