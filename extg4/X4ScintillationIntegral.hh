#pragma once
#include "X4_API_EXPORT.hh"

class G4PhysicsOrderedFreeVector ; 
#include "G4MaterialPropertyVector.hh"

struct X4_API X4ScintillationIntegral
{
    static G4PhysicsOrderedFreeVector* Integral( const G4MaterialPropertyVector* theFastLightVector ) ;
}; 


