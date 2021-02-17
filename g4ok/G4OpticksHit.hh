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
    G4int         sensorIndex ;
    G4int         nodeIndex ;
    G4int         photonIndex ;
    G4int         flag_mask ; 
    G4int         sensor_identifier ; 
    G4bool        is_cerenkov ; 
    G4bool        is_reemission ; 


};


/**
G4OpticksHitExtra
-------------------

This extra hit information is only filled when Opticks::isWayEnabled 
is switched on with the --way option on the embedded Opticks commandline.

t0 
   initial time of the first photon at generation from the genstep 
   obtained some other particle. This time is obtained immediately 
   after generation before starting the "bounce" loop 

boundary_pos
   global frame position of the photon when it crosses a boundary 
   configured by (TODO: lookup how to configure which boundary to record)) 


**/

struct G4OpticksHitExtra
{
    G4ThreeVector boundary_pos ; 
    G4double      boundary_time ;     

    G4double      origin_time ;     
    G4int         origin_trackID ; 
};



