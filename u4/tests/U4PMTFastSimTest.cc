#include "U4RecorderTest.h"
#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"
#include "U4Engine.h"
#include "U4UniformRand.h"

#include "J_PMTFASTSIM_LOG.hh"


struct U4PMTFastSimTest
{
    static G4RunManager* InitRunManager(G4VUserPhysicsList* phy);  
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 
    NP*                        bef ;  

    U4PMTFastSimTest(); 
    void BeamOn(); 
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
    rec(new U4RecorderTest(run)),
    bef(nullptr)
{
}

void U4PMTFastSimTest::BeamOn()
{
    int n = U::GetEnvInt("BeamOn",1) ; 
    // if( n < 0 ) bef = U4UniformRand::Get(1000);    HMM: dont match because the g4states are not restored yet 
    if( n > 0 ) run->BeamOn(n); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    J_PMTFASTSIM_LOG_(0); 

    LOG(info) << "[ " << argv[0] << " " << STime::Now() ; 
    LOG(info) << U4Engine::Desc()  ; 

    SEvt* evt = SEvt::CreateOrLoad() ; 
    if(evt->is_loaded) evt->clear_partial("g4state");  // clear loaded evt but keep g4state

    SEvt::AddTorchGenstep(); 

    U4PMTFastSimTest t ;  
    t.BeamOn(); 
    
    if(evt->is_loaded) evt->setReldir("SEL");  // evt is loaded when rerunning a single photon

    evt->save(); 
    const char* savedir = evt->getSaveDir(); 
    SFastSim_Debug::Save(savedir); 

    U4Recorder* fRecorder = t.rec->fRecorder ; 
    fRecorder->saveRerunRand(savedir); 

    if(t.bef) t.bef->save(savedir, "bef.npy") ; // doesnt match without BeamOn as g4state not restored yet 

    LOG(info) << " savedir " << savedir ;  

    LOG(info) << "] " << argv[0] << " " << STime::Now() ; 
    return 0 ; 
}

