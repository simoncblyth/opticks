#pragma once
#include <string>
#include <sstream>
#include <cstdint>

#include "G4Types.hh"
#include "G4ThreeVector.hh"

/**1
U4Hit 
------

Type used in G4Opticks interface, NB all Geant4 types no Opticks ones.

Note that Opticks does not use the Geant4 system of units approach, it 
picks a standard set of units suited to optical physics and uses
these unless described otherwise. 

* distances (mm)
* wavelength (nm)
* energy (eV)
* time (ns) 

Reasons for this include that Opticks mostly uses float precision only 
resorting to double precision where that is unavoidable. This is 
contrary to Geant4 which uses double precision everywhere. 
Also Opticks compartmentalizes its dependency on Geant4 headers to 
only a few of its highest level sub-packages.

1**/

struct U4Hit 
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
    uint64_t      photonIndex ;
    G4int         flag_mask ; 
    G4int         sensor_identifier ; 
    G4bool        is_cerenkov ; 
    G4bool        is_reemission ; 

    void zero() ; 
    std::string desc() const ; 
};


#include "U4ThreeVector.h"

inline void U4Hit::zero() 
{
    local_position.set(0.,0.,0.); 
    global_position.set(0.,0.,0.); 
    time = 0. ; 
    local_direction.set(0.,0.,0.); 
    global_direction.set(0.,0.,0.); 
    weight = 0. ; 
    local_polarization.set(0.,0.,0.); 
    global_polarization.set(0.,0.,0.); 
    wavelength = 0. ; 
    boundary = 0 ; 
    sensorIndex = 0 ; 
    nodeIndex = 0 ; 
    photonIndex = 0 ; 
    flag_mask = 0 ; 
    sensor_identifier = 0 ; 
    is_cerenkov = false ; 
    is_reemission = false ; 
}


inline std::string U4Hit::desc() const
{
    std::stringstream ss ; 

    ss << "U4Hit::desc" 
       << " lpos " << U4ThreeVector::Desc(local_position) 
       << " gpos " << U4ThreeVector::Desc(global_position) 
       ; 

    std::string s = ss.str(); 
    return s ; 
}




struct U4HitExtra
{
    G4ThreeVector boundary_pos ; 
    G4double      boundary_time ;     

    G4double      origin_time ;     
    G4int         origin_trackID ; 
};




