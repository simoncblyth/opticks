#include "U4RecorderTest.h"
#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"

#include "U4Engine.h"
#include "U4UniformRand.h"

#include "InstrumentedG4OpBoundaryProcess.hh"
#include "junoPMTOpticalModel.hh"

#include "J_PMTFASTSIM_LOG.hh"

struct U4PMTFastSimTest
{
    static G4RunManager* InitRunManager(G4VUserPhysicsList* phy);  
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 

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
    rec(new U4RecorderTest(run))
{
}

void U4PMTFastSimTest::BeamOn()
{
    run->BeamOn(U::GetEnvInt("BeamOn",1)); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    J_PMTFASTSIM_LOG_(0); 

    const char* ekey = "hama_UseNaturalGeometry" ; 
    int eval = SSys::getenvint(ekey, 0 );  
    LOG(info) 
        << "[ " << argv[0] << " " << STime::Now() 
        << " ekey " << ekey 
        << " eval " << eval 
        ;

    LOG(info) << U4Engine::Desc()  ; 

    SEvt* evt = SEvt::CreateOrLoad() ; 
    bool is_loaded = evt->is_loaded ;  // true when rerunning as single photon
    if(is_loaded) 
    {
        evt->clear_partial("g4state");  // clear loaded evt but keep g4state
        std::string reldir = U::FormName("SEL", eval, nullptr ); 
        LOG(info) << " reldir " << reldir ; 
        evt->setReldir(reldir.c_str());
    }

    SEvt::AddTorchGenstep(); 

    U4PMTFastSimTest t ;  
    t.BeamOn(); 


    evt->save(); 
    const char* savedir = evt->getSaveDir(); 
    SFastSim_Debug::Save(savedir); 
    junoPMTOpticalModel::Save(savedir); 
    InstrumentedG4OpBoundaryProcess::Save(savedir); 

    U4Recorder* fRecorder = t.rec->fRecorder ; 
    fRecorder->saveRerunRand(savedir); 
    LOG(info) << " savedir " << savedir ;  

    LOG(info) << "] " << argv[0] << " " << STime::Now() ; 
    return 0 ; 
}

