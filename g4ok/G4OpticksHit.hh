#pragma once

#include "G4Types.hh"
#include "G4ThreeVector.hh"

/**1
G4OpticksHit 
-------------

Type used in G4Opticks interface, NB all Geant4 types no Opticks ones.

1**/

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

/**2
G4OpticksHitExtra
-------------------

This extra hit information is only filled when ``Opticks::isWayEnabled`` 
is switched on with the ``--way`` option on the embedded Opticks commandline.

``boundary_pos`` ``boundary_time``
   global frame position and time of a photon when it crosses a boundary 
   selected by **geospecific** commandline options.

``origin_time`` 
   initial time of the first photon at generation from the genstep 
   obtained some other particle. This time is obtained immediately 
   after generation before starting the "bounce" loop 

``origin_trackID``
   non-optical parent G4Track::GetTrackID recorded into genstep at ``0*4+1``

2**/

struct G4OpticksHitExtra
{
    G4ThreeVector boundary_pos ; 
    G4double      boundary_time ;     

    G4double      origin_time ;     
    G4int         origin_trackID ; 
};

/**3

Examples of the **geospecific** options for different geometries::

   --boundary MineralOil///Acrylic --pvname /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 
   --boundary Water///Acrylic --pvname pInnerWater 
   --boundary -Water///Acrylic     # negating the sign of the boundary index 

Such options are used via GGeo::getSignedBoundary Opticks::getBoundary GGeo::getFirstNodeIndexForPVName Opticks::getPVName
in OGeo::initWayControl to set **way_control** in the OptiX GPU context.

For each boundary encountered in the GPU propagation in oxrap/cu/generate.cu a match with the 
boundary and/or pvname is checked and when found results in the setting of ``boundary_pos`` ``boundary_time``.

**currently only the signed boundary is being checked, TODO: make this configurable without changing code**


3**/



