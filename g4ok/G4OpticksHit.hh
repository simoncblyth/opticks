#pragma once

#include "G4Types.hh"
#include "G4ThreeVector.hh"

/**
G4OpticksHit 
-------------

Type used in G4Opticks interface, NB all Geant4 types no Opticks ones.

**/

struct G4OpticksHit 
{
    G4ThreeVector local_position ; 
    G4ThreeVector global_position ; 
    G4double      time ; 
    G4ThreeVector local_direction ; 
    G4ThreeVector global_direction ; 
    G4double      weight ; 
    G4ThreeVector local_polarization ; 
    G4ThreeVector global_polarization ; 
    G4double      wavelength ; 
    G4int         boundary ;
    G4int         sensor_index ;
    G4int         node_index ;
    G4int         photon_index ;
    G4int         flag_mask ; 
    G4int         sensor_identifier ; 
    G4bool        is_cerenkov ; 
    G4bool        is_reemission ; 
};




