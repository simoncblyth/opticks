/**
GeoChainTest.cc : testing the full chain of geometry conversions
===================================================================

The geometry to create is controlled by the name string obtained from envvar *GEOCHAINTEST* 

1. creates G4VSolid directly here or using functionality from other libs such as j/PMTSim(ZSolid)
2. invokes GeoChain::convert

   * (x4) X4PhysicalVolume::ConvertSolid : G4VSolid -> nnode -> GMesh/GPts
   * (this) using placeholder GVolume the GMesh is added to a test GGeo
   * (cg) CSG_GGeo_Convert GGeo -> CSGFoundry  

3. save the CSGFoundry geometry under directory named by the *GEOCHAINTEST* name


Subsequently can render this geometry, eg with CSGOptiX/cxs.sh using 
just the path to the CSGFoundry directory. 


TODO : expand from single solid conversions to G4VPhysicalVolume of small collections of solids (eg PMT)
----------------------------------------------------------------------------------------------------------

Note that CSG_GGeo is already able to convert full geometries, so all functionality 
is already available it just needs to be used from this testing environment.
This will probably entail modifications to make it easier to do so (eg rearranging functionality
into static methods). 

**/

#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "GeoChain.hh"
#include "GeoMaker.hh"

#include "G4VSolid.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

const G4VSolid* const make_PMTSim(const char* name, std::string& meta)
{
    std::stringstream ss ; 
    ss << "creator:make_PMTSim" << std::endl ; 
    const G4VSolid* solid = nullptr ;  
#ifdef WITH_PMTSIM
    LOG(info) << "[ PMTSim::GetSolid name " << name ; 
    solid = PMTSim::GetSolid(name) ;   // for zcut include integer in name eg "PMTSim_Z-400" 
    LOG(info) << "] PMTSim::GetSolid GetName " << solid->GetName()  ; 
#endif
    meta = ss.str();   
    return solid ; 
}

const G4VSolid* const make_solid(const char* name, std::string& meta)
{
    LOG(info) << "[ " << name ; 
    const G4VSolid* solid = nullptr ; 
    if(strcmp(name,"default") == 0)                      solid = GeoMaker::make_default(name);  
    if(strcmp(name,"AdditionAcrylicConstruction") == 0 ) solid = GeoMaker::make_AdditionAcrylicConstruction(name); 
    if(strcmp(name,"BoxMinusTubs0") == 0 )               solid = GeoMaker::make_BoxMinusTubs0(name); 
    if(strcmp(name,"BoxMinusTubs1") == 0 )               solid = GeoMaker::make_BoxMinusTubs1(name); 
    if(SStr::StartsWith(name, "PMTSim"))                 solid = make_PMTSim(name, meta) ;  
    assert(solid); 
    //G4cout << *solid << G4endl ; 
    LOG(info) << "] " << name ; 
    return solid ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    const char* name_default = "AdditionAcrylicConstruction"  ; 
    const char* name = SSys::getenvvar("GEOCHAINTEST", name_default ); 

    std::string meta ; 
    const G4VSolid* const solid = make_solid(name, meta);   

    const char* argforced = "--allownokey" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 
    //for(int lvIdx=-1 ; lvIdx < 10 ; lvIdx+= 1 ) LOG(info) << " lvIdx " << lvIdx << " ok.isX4TubsNudgeSkip(lvIdx) " << ok.isX4TubsNudgeSkip(lvIdx)  ; 

    GeoChain chain(&ok); 
    chain.convert(solid, meta);  
    chain.save(name); 

    return 0 ; 
}
