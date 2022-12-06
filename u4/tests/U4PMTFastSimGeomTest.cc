#include "OPTICKS_LOG.hh"
#include "U4VolumeMaker.hh"
#include "G4VPhysicalVolume.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4VPhysicalVolume* pv = const_cast<G4VPhysicalVolume*>(U4VolumeMaker::PV());  // sensitive to GEOM envvar 

    const G4String& pv_name = pv->GetName() ; 

    LOG(info) 
        << " pv " << pv 
        << " pv_name " << pv_name
        ; 

    return 0 ; 
}
