/**
U4SimulateTest.cc ( formerly U4PMTFastSimTest.cc)
======================================================

Most of the Geant4 setup happens on instanciating U4RecorderTest  
from U4RecorderTest.h

**/


#include "U4RecorderTest.h"    


#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"
#include "SEventConfig.hh"

#include "U4Engine.h"
#include "U4UniformRand.h"

#include "InstrumentedG4OpBoundaryProcess.hh"
#include "G4Material.hh"

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


    int g4state_rerun_id = SEventConfig::G4StateRerun(); 
    bool rerun = g4state_rerun_id > -1 ;  
    const char* seldir = U::FormName( "SEL", VERSION, nullptr ); 
    const char* alldir = U::FormName( "ALL", VERSION, nullptr ); 
    const char* alldir0 = U::FormName( "ALL", 0, nullptr ); 


    LOG(info) 
        << " g4state_rerun_id " << g4state_rerun_id 
        << " alldir " << alldir 
        << " alldir0 " << alldir0 
        << " seldir " << seldir 
        << " rerun " << rerun
        ; 

    SEvt* evt = nullptr ; 
    if(rerun == false)
    {
        evt = SEvt::Create();    
        evt->setReldir(alldir);
    }  
    else
    {
        evt = SEvt::Load(alldir0) ;
        evt->clear_partial("g4state");  // clear loaded evt but keep g4state 
        evt->setReldir(seldir);
        // when rerunning have to load states from alldir0 and then change reldir to save into seldir
    }
    // HMM: note how reldir at object rather then static level is a bit problematic for loading 


    SEvt::AddTorchGenstep(); 

    U4SimulateTest t ;  
    t.BeamOn(); 

    evt->save(); 
    const char* savedir = evt->getSaveDir(); 

    SFastSim_Debug::Save(savedir); 

#ifdef WITH_PMTFASTSIM
    junoPMTOpticalModel::Save(savedir); 
    InstrumentedG4OpBoundaryProcess::Save(savedir); 
#endif

    U4Recorder* fRecorder = t.rec->fRecorder ; 
    fRecorder->saveRerunRand(savedir); 
    LOG(info) << " savedir " << savedir ;  

    LOG(info) << "] " << argv[0] << " " << STime::Now() << " VERSION " << VERSION ; 


    G4Material* Pyrex = G4Material::GetMaterial("Pyrex"); 
    G4Material* Vacuum = G4Material::GetMaterial("Vacuum"); 

    LOG(info) << " Pyrex " << ( Pyrex ? "Y" : "N" ) ; 
    LOG(info) << " Vacuum " << ( Vacuum ? "Y" : "N" ) ; 

    return 0 ; 
}

