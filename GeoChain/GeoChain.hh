#pragma once

/**
GeoChain (considered naming it CSG_G4 in analogy to CSG_GGeo)
=================================================================

Raison-d'etre of *GeoChain* is to perform the full chain of geometry conversions in a single executable. 

The primary motivation is to provide fast geometry iteration to investigate issues. 
Aim to be able to edit Geant4 C++ solid definition then run a single executable to get a 
rendering of the geometry including 2D cross sections. Actually while are still 
in OptiX transition it is expedient to keep rendering in a separate package and executable. 

Note that this means will need to depend on most everything and the kitchensink, but
that is OK as the aim of this package is narrow. The product is the executable not the API. 

Stages of the Geometry Chain
------------------------------

1. Geant4 C++ geometry definition
2. X4PhysicalVolume::ConvertSolid orchestrates G4VSolid -> nnode/NCSG/GMesh 

   * X4Solid::Convert converts G4VSolid into npy/nnode tree
   * NTreeProcess<nnode>::Process balances the nnode tree when that is configured
   * NCSG::Adopt wrap nnode tree enabling it to travel 
   * X4Mesh::Convert converts G4VSolid into GMesh which has above created NCSG associated 

3. CSG_ 


While the initial focus on single G4VSolid shapes, do not want to 
add new code to support this, want to use the standard code path 
as much as possible. That will probably mean using GMergedMesh and 
combi GParts even when it contains only a single GMesh.

NB : AVOID WRITING NEW CODE, INSTEAD ADJUST EXISTING API (eg ADD STATIC METHODS) 
TO MAKE IT USABLE FROM HERE 

**/

#include "GEOCHAIN_API_EXPORT.hh"
#include "plog/Severity.h"
#include <string>

class G4VSolid ; 
class G4VPhysicalVolume ; 

class Opticks ; 
class GGeo ; 
class GMesh ; 
class GVolume ; 
struct CSGFoundry ; 
struct nnode ; 

struct GEOCHAIN_API GeoChain
{
    static const plog::Severity LEVEL ; 
    static const char* BASE ; 

    Opticks* ok ; 
    GGeo* ggeo ; 
    GMesh* mesh ;
    GVolume* volume ;
    CSGFoundry* fd ;  
    int lvIdx ; 
    int soIdx ; 
 
    GeoChain(Opticks* ok ); 
    
    void init(); 
    void convertSolid(const G4VSolid*          so, std::string& meta ); 
    void convertNodeTree( nnode*             root ); 
    void convertPV(   const G4VPhysicalVolume* pv ); 
    void convertMesh(GMesh* mesh ) ; 
    void convertName(const char* geom ); 

    void save(const char* name, const char* base=nullptr) const ; 
}; 


