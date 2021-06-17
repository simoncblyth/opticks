#pragma once


/**
X4MaterialWater : special case fallback handling of "Water" G4Material without a RAYLEIGH scattering length property
======================================================================================================================

When a G4Material with name "Water" does not have a kRAYLEIGH/"RAYLEIGH" property the method 
G4OpRayleigh::BuildPhysicsTable invokes G4OpRayleigh::CalculateRayleighMeanFreePaths 
which calculates the scattering lengths based on the RINDEX property which must be present. 
The calculated rayleigh scattering lengths are unfortunately not inserted into the 
material they are instead inserted into the G4OpRayleigh::thePhysicsTable

That is problematic for the Opticks geometry conversion which acts on the G4Material. 
When the Opticks "x4" conversion finds a material without a RAYLEIGH property it adopts 
a constant placeholder default value for the rayleigh scattering length.  

The resulting difference in scattering lengths caused a discrepancy 
in the amount of scattering observed in water between Geant4 and Opticks. 
Geant4 scattered more because the Opticks default scattering length is very large.
 
In order to avoid this discrepancy it is necessary for Opticks to perform some 
special case handling of G4Material named "Water" in order to match the Geant4 fallback behaviour 
for "Water" materials without a "RAYLEIGH" property.

This struct X4MaterialWater is intended to remedy the situation using its own G4OpRayleigh
process to perform the fallback calculation when there is no RAYLEIGH property in the Water G4Material
and add the resulting scattering lengths to the Water G4Material in order to enable the 
Opticks conversion to benefit from the same fallback calculation as Geant4.


Note annoying typedef::

   typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector

**/

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include "G4types.hh"

class G4Material ; 
class G4MaterialPropertiesTable ; 
class G4PhysicsOrderedFreeVector ; 

struct X4_API X4MaterialWater
{
    static const plog::Severity  LEVEL ; 
    static bool IsApplicable();   // returns true when "Water" G4Material has RINDEX but not RAYLEIGH
    static G4PhysicsOrderedFreeVector* GetRAYLEIGH(); 
    static G4PhysicsOrderedFreeVector* GetRINDEX(); 
    static G4PhysicsOrderedFreeVector* GetProperty(const G4int index); 

    G4Material*                  Water ; 
    G4MaterialPropertiesTable*   WaterMPT ;  
    G4PhysicsOrderedFreeVector*  rayleigh0 ; // from the material, possibly null         
    G4PhysicsOrderedFreeVector*  rayleigh ;  // from the material if present otherwise calculated from RINDEX     
    
    X4MaterialWater(); 
    void init(); 

    void dump() const ; 
    void rayleigh_scan() const ; 
    void rayleigh_scan2() const ; 
    void changeRayleighToMidBin(); // just used for X4MaterialWaterTest

    G4double GetMeanFreePath( G4double energy ) const ; 
}; 


