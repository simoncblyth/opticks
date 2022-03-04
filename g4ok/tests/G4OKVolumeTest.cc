#include <cassert>
#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "G4PVPlacement.hh"
#include "Opticks.hh"
#include "G4Opticks.hh"
#include "G4OpticksRecorder.hh"  
#include "G4OpticksHit.hh"
#include "OpticksFlags.hh"

#include "X4VolumeMaker.hh"


struct G4OKVolumeTest
{
    G4Opticks* g4ok ; 

    G4OKVolumeTest(); 
    virtual ~G4OKVolumeTest();

};


G4OKVolumeTest::G4OKVolumeTest()
    :
    g4ok(new G4Opticks)
{
}

G4OKVolumeTest::~G4OKVolumeTest()
{
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const char* geom_default = "nnvtBodyLogWrapLV" ; 
    const char* geom_default = "JustOrbGrid" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default );  

    G4VPhysicalVolume* pv = X4VolumeMaker::Make(geom) ; 

    if( pv == nullptr )
    {
        LOG(fatal) << " failed to get GEOM [" << geom << "]" ; 
        return 0 ; 
    }

    LOG(info) << " pv " << pv->GetName() ; 


    G4OKVolumeTest t ; 
    t.g4ok->setGeometry(pv); 

    Opticks* ok = Opticks::Get() ;
    ok->reportKey(argv[0]);   // TODO: do this standardly within setGeometry 

    return 0 ; 
}
