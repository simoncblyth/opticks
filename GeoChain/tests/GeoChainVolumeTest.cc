/**
GeoChainVolumeTest.cc : testing the full chain of geometry conversions
=========================================================================

The volume to create is controlled by the name string obtained from envvar *GEOCHAINTEST* 

**/

#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "GeoChain.hh"

#include "G4VPhysicalVolume.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    const char* name_default = "body_phys"  ; 
    const char* name = SSys::getenvvar("GEOM", name_default ); 

    const G4VPhysicalVolume* pv = nullptr ; 
#ifdef WITH_PMTSIM
    pv = PMTSim::GetPV(name);  
#endif
    assert( pv ); 

    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    // see notes/issues/PMT_body_phys_bizarre_innards_confirmed_fixed_by_using_gparts_transform_offset_option.rst
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    std::stringstream ss ; 
    ss << "creator:GeoChainVolumeTest" << std::endl ; 
    ss << "name:" << name << std::endl ; 
    ss << "gparts_transform_offset:" << ( ok.isGPartsTransformOffset() ? "YES" : "NO" ) << std::endl ; 
    std::string meta = ss.str(); 

    GeoChain chain(&ok); 
    chain.convertPV(pv, meta);  

    chain.save(GeoChain::BASE, name); 

    return 0 ; 
}
