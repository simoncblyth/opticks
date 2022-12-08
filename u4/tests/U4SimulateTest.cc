/**
U4SimulateTest.cc ( formerly U4PMTFastSimTest.cc)
======================================================


**/

#include "U4RecorderTest.h"
#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"
#include "U4Engine.h"
#include "U4UniformRand.h"

#include "InstrumentedG4OpBoundaryProcess.hh"

#ifdef WITH_PMTFASTSIM
#include "junoPMTOpticalModel.hh"
#include "J_PMTFASTSIM_LOG.hh"
#endif


struct U4SimulateTest
{
    static G4RunManager* InitRunManager(G4VUserPhysicsList* phy);  
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 

    U4SimulateTest(); 
    void BeamOn(); 
    virtual ~U4SimulateTest(){ delete rec ; }
};

G4RunManager* U4SimulateTest::InitRunManager(G4VUserPhysicsList* phy)
{
    G4RunManager* run = new G4RunManager ; 
    run->SetUserInitialization(phy) ; 
    return run ; 
}

U4SimulateTest::U4SimulateTest()
    :
    phy((G4VUserPhysicsList*)new U4Physics),
    run(InitRunManager(phy)),
    rec(new U4RecorderTest(run))
{
}

void U4SimulateTest::BeamOn()
{
    run->BeamOn(U::GetEnvInt("BeamOn",1)); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
#ifdef WITH_PMTFASTSIM
    J_PMTFASTSIM_LOG_(0); 
#endif

    int VERSION = SSys::getenvint("VERSION", 0 );  
    LOG(info) << "[ " << argv[0] << " " << STime::Now() << " VERSION " << VERSION ; 
    LOG(info) << U4Engine::Desc()  ; 

    SEvt* evt = SEvt::CreateOrLoad() ; 

    bool rerun = evt->is_loaded ;  
    if(rerun) evt->clear_partial("g4state");  // clear loaded evt but keep g4state 

    std::string reldir = U::FormName( rerun ? "SEL" : "ALL" , VERSION, nullptr ); 
    LOG(info) << " reldir " << reldir << " rerun " << rerun ; 
    evt->setReldir(reldir.c_str());

    SEvt::AddTorchGenstep(); 

    U4SimulateTest t ;  
    t.BeamOn(); 

    evt->save(); 
    const char* savedir = evt->getSaveDir(); 

    SFastSim_Debug::Save(savedir); 

#ifdef WITH_PMTFASTSIM
    junoPMTOpticalModel::Save(savedir); 
#endif
    InstrumentedG4OpBoundaryProcess::Save(savedir); 

    U4Recorder* fRecorder = t.rec->fRecorder ; 
    fRecorder->saveRerunRand(savedir); 
    LOG(info) << " savedir " << savedir ;  

    LOG(info) << "] " << argv[0] << " " << STime::Now() << " VERSION " << VERSION ; 
    return 0 ; 
}

