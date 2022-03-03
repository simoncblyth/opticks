#include <cassert>
#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "G4PVPlacement.hh"
#include "Opticks.hh"
#include "G4Opticks.hh"
#include "G4OpticksRecorder.hh"  
#include "G4OpticksHit.hh"
#include "OpticksFlags.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif


struct G4OKPMTSimTest
{
    G4Opticks* g4ok ; 

    G4OKPMTSimTest(); 
    virtual ~G4OKPMTSimTest();

};


G4OKPMTSimTest::G4OKPMTSimTest()
    :
    g4ok(new G4Opticks)
{
}

G4OKPMTSimTest::~G4OKPMTSimTest()
{
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom_default = "nnvtBodyLogWrapLV" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default );  

    G4VPhysicalVolume* pv = nullptr ; 

#ifdef WITH_PMTSIM
    pv = PMTSim::GetPV(geom);
#else
#endif

    if( pv == nullptr )
    {
        LOG(fatal) << " failed to get GEOM [" << geom << "]" ; 
        return 0 ; 
    }

    LOG(info) << " pv " << pv->GetName() ; 


    G4OKPMTSimTest t ; 
    t.g4ok->setGeometry(pv); 

    Opticks* ok = Opticks::Get() ;
    ok->reportKey(argv[0]);   // TODO: do this standardly within setGeometry 

    return 0 ; 
}
