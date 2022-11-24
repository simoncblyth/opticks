#include "U4RecorderTest.h"
#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"
#include "U4Engine.h"

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

    LOG(info) << "[ " << argv[0] << " " << STime::Now() ; 
    LOG(info) << U4Engine::Desc()  ; 

    SEvt* evt = SEvt::CreateOrLoad() ; 

    SEvt::AddTorchGenstep(); 

    U4PMTFastSimTest t ;  

    SFastSim_Debug::Save("/tmp/SFastSim_Debug" ); 

    
    if(evt->is_loaded) 
    {
        LOG(info) << " not saving as evt->is_loaded " ; 
        // TODO: save under a different reldir 
        // the default reldir is ALL
        // event loading is done for rerunning 
        // so the reldir needs to reflect that 
    }
    else
    {
        evt->save(); 
    }

    LOG(info) << "] " << argv[0] << " " << STime::Now() ; 
    return 0 ; 
}

