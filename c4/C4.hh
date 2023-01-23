#pragma once
#include <string>

#include "C4_API_EXPORT.hh"

class G4VPhysicalVolume ; 
struct SSim ; 
struct CSGFoundry ; 

struct C4_API C4
{
    SSim*       si ; 
    CSGFoundry* fd ; 

    static CSGFoundry* Translate(const G4VPhysicalVolume* const top ); 

    C4(const G4VPhysicalVolume* const top); 
    void init(); 

    std::string desc() const ; 
}; 





