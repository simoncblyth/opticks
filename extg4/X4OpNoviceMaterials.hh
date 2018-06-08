#pragma once


#include "X4_API_EXPORT.hh"

class G4Material ;

struct X4_API X4OpNoviceMaterials
{
    G4Material* air ; 
    G4Material* water ;

    X4OpNoviceMaterials();
}; 
 



