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

3. CSG 


While the initial focus on single G4VSolid shapes, do not want to 
add new code to support this, want to use the standard code path 
as much as possible. That will probably mean using GMergedMesh and 
combi GParts even when it contains only a single GMesh.

NB : AVOID WRITING NEW CODE, INSTEAD ADJUST EXISTING API (eg ADD STATIC METHODS) 
TO MAKE IT USABLE FROM HERE 

**/


#include "G4Orb.hh"

#include "SSys.hh"
#include "OPTICKS_LOG.hh"

#include "Opticks.hh"

#include "GMesh.hh"
#include "GGeo.hh"
#include "X4PhysicalVolume.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"


struct GeoChain
{
    Opticks* ok ; 
    GGeo* ggeo ; 
    GMesh* mesh ;
 
    GeoChain(Opticks* ok ); 
    void convert(const G4VSolid* const solid); 
}; 

GeoChain::GeoChain(Opticks* ok_)
    :
    ok(ok_), 
    ggeo(new GGeo(ok, true)),  // live=true to initLibs and not load from cache
    mesh(nullptr)
{
}

void GeoChain::convert(const G4VSolid* const solid)
{
    int lvIdx = 0 ; 
    int soIdx = 0 ; 
    std::string lvname = solid->GetName(); 

    mesh = X4PhysicalVolume::ConvertSolid(ok, lvIdx, soIdx, solid, lvname ) ; 
    LOG(info) << " mesh " << mesh ; 

    ggeo->add(mesh); 
    ggeo->deferredCreateGParts(); 
    ggeo->prepareVolumes(); 

    // hmm everything is based on GMergedMesh : so the above do little without 
    // having any volumes... need an artifical one ?

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* argforced = "--noinstanced" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    GeoChain gc(&ok); 

    G4Orb s("orb", 100.) ; 
    G4cout << s << G4endl ; 



    gc.convert(&s);  


    CSGFoundry fd ; 
    CSG_GGeo_Convert conv(&fd, gc.ggeo) ; 
    conv.convert(); 

    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/GeoChain/default" );
    const char* rel = "CSGFoundry" ; 

    fd.write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 

    CSGFoundry* lfd = CSGFoundry::Load(cfbase, rel);  // load foundary and check identical bytes
    assert( 0 == CSGFoundry::Compare(&fd, lfd ) );  


    return 0 ; 
}
