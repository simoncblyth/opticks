/**
GeoChainSolidTest.cc : testing the full chain of geometry conversions for single solids
==========================================================================================

The solid to create is controlled by the name string obtained from envvar *GEOM* 

1. creates G4VSolid directly here or using functionality from other libs such as j/PMTSim(ZSolid)
2. invokes GeoChain::convert

   * (x4) X4PhysicalVolume::ConvertSolid : G4VSolid -> nnode -> GMesh/GPts

     * X4Solid::Convert converts G4VSolid into npy/nnode tree
     * NTreeProcess<nnode>::Process balances the nnode tree when that is configured
     * NCSG::Adopt wrap nnode tree enabling it to travel 
     * X4Mesh::Convert converts G4VSolid into GMesh which has above created NCSG associated 

   * (this) using placeholder GVolume the GMesh is added to a test GGeo
   * (cg) CSG_GGeo_Convert GGeo -> CSGFoundry  

3. saves geometry to $TMP/GeoChain/$GEOM/CSGFoundry/ 

Subsequently can render this geometry, eg with CSGOptiX/cxs.sh using 
just the path to the CSGFoundry directory. 


**/

#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "GeoChain.hh"

#include "G4VSolid.hh"
#include "X4Intersect.hh"
#include "X4SolidMaker.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif


const G4VSolid* GetSolid(const char* geom)
{
    const G4VSolid* solid = nullptr ; 
    if(X4SolidMaker::CanMake(geom))
    {
        solid = X4SolidMaker::Make(geom); 
    }
    else
    {
#ifdef WITH_PMTSIM
        solid = PMTSim::GetSolid(geom); 
#endif
    }
    return solid ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom = SSys::getenvvar("GEOM", "AdditionAcrylicConstruction" ); 

    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    GeoChain chain(&ok); 

    const G4VSolid* solid = GetSolid(geom);  
    if( solid )
    {
        chain.convertSolid(solid);  
    }
    else
    {
        chain.convertName(geom);         
    }

    chain.save(geom); 

    return 0 ; 
}


