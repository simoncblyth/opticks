#include <cassert>
#include <csignal>
#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "Opticks.hh"
#include "G4Opticks.hh"
#include "G4VPhysicalVolume.hh"
#include "X4VolumeMaker.hh"

G4VPhysicalVolume* PV()
{
    //const char* geom_default = "nnvtBodyLogWrapLV" ; 
    const char* geom_default = "JustOrbGrid" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default );  

    G4VPhysicalVolume* pv = X4VolumeMaker::Make(geom) ; 
    if( pv == nullptr )
    {
        LOG(fatal) << "X4VolumeMaker::Make FAILED for GEOM [" << geom << "]" ; 
        std::raise(SIGINT); 
        return nullptr ; 
    }
    LOG(info) << " pv " << pv->GetName() ; 
    return pv ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4VPhysicalVolume* pv = PV(); 

    G4Opticks* g4ok = new G4Opticks ; 
    g4ok->setGeometry(pv); 

    // HMM: GParts creation makes more sense on loading just prior to CSG_GGeo, 
    // as then --gparts_transform_offset will be set 
    //g4ok->saveGParts(); 

    const char* bin = argv[0]; 
    Opticks* ok = Opticks::Get() ;
    ok->reportKey(bin);   // TODO: do this standardly within setGeometry 
    ok->writeGeocacheScript(bin); 

    return 0 ; 
}
