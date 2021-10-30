#pragma once
/**
X4PhysicsOrderedFreeVector.hh
===============================

Follow suggestion of Soon Yung Jun to workaround the dropping of 
G4PhysicsOrderedFreeVector from Geant4 11.

The methods of G4PhysicsOrderedFreeVector have been consolidated into 
into G4PhysicsFreeVector with Geant4 11.

NB the former X4PhysicsOrderedFreeVector.hh was renamed to X4Array.hh


**/
#include "G4Version.hh"

#if G4VERSION_NUMBER < 1100
#include "G4PhysicsOrderedFreeVector.hh"
#else
#include "G4PhysicsFreeVector.hh"
typedef G4PhysicsFreeVector G4PhysicsOrderedFreeVector ;   
// create "G4PhysicsOrderedFreeVector" type identifier as alias for "G4PhysicsFreeVector" 
// as G4 11 dropped it
#endif

