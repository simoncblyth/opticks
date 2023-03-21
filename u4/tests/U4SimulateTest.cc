/**
U4SimulateTest.cc ( formerly U4PMTFastSimTest.cc)
======================================================

Most of the Geant4 setup happens on instanciating U4App from U4App.h

**/

#include "ssys.h"
#include "U4App.h"    
#include "U4AppTest.h"    

#include "STime.hh"
#include "SEvt.hh"
#include "SFastSim_Debug.hh"
#include "SEventConfig.hh"

#include "U4Engine.h"
#include "U4UniformRand.h"
#include "U4VolumeMaker.hh"
#include "U4Recorder.hh"


#ifdef WITH_PMTSIM

#include "J_PMTSIM_LOG.hh"
#include "PMTSim.hh"
#include "junoPMTOpticalModel.hh"

#elif WITH_PMTFASTSIM

#include "junoPMTOpticalModel.hh"
#include "J_PMTFASTSIM_LOG.hh"

#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_PMTSIM
    J_PMTSIM_LOG_(0); 
#elif WITH_PMTFASTSIM
    J_PMTFASTSIM_LOG_(0); 
#endif

    int VERSION = SSys::getenvint("VERSION", 0 );  
    LOG(info) << "[ " << argv[0] << " " << STime::Now() << " VERSION " << VERSION ; 

    SEvt* evt = SEvt::HighLevelCreate(); 

    U4AppTest t ;  
    t.BeamOn(); 

    evt->save(); 
    const char* savedir = evt->getSaveDir(); 


    SFastSim_Debug::Save(savedir); 
#if defined(WITH_PMTSIM) && defined(POM_DEBUG)
    PMTSim::ModelTrigger_Debug_Save(savedir) ; 
    U4VolumeMaker::SaveTransforms(savedir) ; 
#else
    LOG(info) << "not-POM_DEBUG  "  ; 
#endif

    U4Recorder::SaveMeta(savedir); 

    LOG(info) << "] " << argv[0] << " " << STime::Now() << " VERSION " << VERSION << " savedir " << savedir ; 
    return 0 ; 
}

