#pragma once

#include "X4_API_EXPORT.hh"

class G4Material ; 
class G4ParticleDefinition ; 
class G4OpRayleigh ; 
class G4PhysicsTable ; 
class G4PhysicsVector ; 

/**
X4OpRayleigh
=============

Depends on the existance of a G4Material named "Water".

X4OpRayleigh::WaterScatteringLength()
   If the G4Material named "Water" has a RAYLEIGH property then that is returned. 
   otherwise if it does have RINDEX then a fallback calulation from G4OpRayleigh is applied 
   and the result fished out of G4OpRayleigh::thePhysicsTable is returned. 

**/

struct X4_API X4OpRayleigh 
{
    static G4PhysicsVector* WaterScatteringLength(); 
    static G4PhysicsVector* GetFromPhysicsTable(const G4OpRayleigh* proc, size_t index ); 

    G4Material*                  Water ; 
    size_t                       WaterIndex ; 
    G4ParticleDefinition*        OpticalPhoton ; 
    G4OpRayleigh*                RayleighProcess ; 
    G4PhysicsTable*              thePhysicsTable; 
    G4PhysicsVector*             rayleigh ;  

    X4OpRayleigh(); 

    void init(); 
    void dump() const ;

}; 
