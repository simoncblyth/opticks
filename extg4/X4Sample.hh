#pragma once

class G4VPhysicalVolume ; 

#include "X4_API_EXPORT.hh"

struct X4_API X4Sample
{
     static G4VPhysicalVolume* Sample(char c);
     static G4VPhysicalVolume* Simple(char c); 
     static G4VPhysicalVolume* OpNovice();
};




