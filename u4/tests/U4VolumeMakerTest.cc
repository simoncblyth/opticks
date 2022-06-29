#include "OPTICKS_LOG.hh"
#include "U4VolumeMaker.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << U4VolumeMaker::Desc() ; 

    G4VPhysicalVolume* pv = U4VolumeMaker::Make();  
    LOG(info) << " pv " << pv ;  

    return 0 ; 
}
