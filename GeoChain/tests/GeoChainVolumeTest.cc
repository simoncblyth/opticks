/**
GeoChainVolumeTest.cc : testing the full chain of geometry conversions
=========================================================================

The volume to create is controlled by the name string obtained from envvar *GEOCHAINTEST* 

TODO : expand from single solid conversions to G4VPhysicalVolume of small collections of solids (eg PMT)

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

#include "G4VPhysicalVolume.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

const G4VPhysicalVolume* const make_PMTSim_PV(const char* name, std::string& meta)
{
    std::stringstream ss ; 
    ss << "creator:GeoChainVolumeTest/make_PMTSimVolume" << std::endl ; 
    const G4VPhysicalVolume* pv = nullptr ;  
#ifdef WITH_PMTSIM
    LOG(info) << "[ make_PMTSim_PV name " << name ; 
    pv = PMTSim::GetPV(name) ;   
    LOG(info) << "] make_PMTSim_PV GetName " << pv->GetName()  ; 
#endif
    meta = ss.str();   
    return pv ; 
}

const G4VPhysicalVolume* make_PV(const char* name, std::string& meta)
{
    LOG(info) << "[ " << name ; 
    const G4VPhysicalVolume* pv = nullptr ; 
    if(SStr::StartsWith(name, "PMTSim"))  pv = make_PMTSim_PV(name, meta) ;  
    assert(pv); 
    LOG(info) << "] " << name ; 
    return pv ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    const char* name_default = "PMTSimLV"  ; 
    const char* name = SSys::getenvvar("GEOCHAINTEST", name_default ); 

    std::string meta ; 
    const G4VPhysicalVolume* pv = make_PV(name, meta);   
    assert( pv ); 

    const char* argforced = "--allownokey" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 
    GeoChain chain(&ok); 
    chain.convertPV(pv, meta);  
    chain.save(name); 

    return 0 ; 
}
