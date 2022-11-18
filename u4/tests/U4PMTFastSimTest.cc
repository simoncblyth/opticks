#include "U4RecorderTest.h"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"

#include "J_PMTFASTSIM_LOG.hh"


struct U4PMTFastSimTest
{
    static G4RunManager* InitRunManager(G4VUserPhysicsList* phy);  
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 
    U4PMTFastSimTest(); 
    virtual ~U4PMTFastSimTest(){ delete rec ; }
};

G4RunManager* U4PMTFastSimTest::InitRunManager(G4VUserPhysicsList* phy)
{
    G4RunManager* run = new G4RunManager ; 
    run->SetUserInitialization(phy) ; 
    return run ; 
}

U4PMTFastSimTest::U4PMTFastSimTest()
    :
    phy((G4VUserPhysicsList*)new U4Physics),
    run(InitRunManager(phy)),
    rec(new U4RecorderTest(run))
{
    run->BeamOn(U::GetEnvInt("BeamOn",1)); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    J_PMTFASTSIM_LOG_(0); 

    SEvt evt ; 
    SEvt::AddTorchGenstep(); 

    U4PMTFastSimTest t ;  

    SFastSim_Debug::Save("/tmp/SFastSim_Debug" ); 

    return 0 ; 
}

