/**
U4SimulateTest.cc
==================

All the Geant4 setup happens in U4App::Create from U4App.h

**/

#include "ssys.h"
#include "U4App.h"    
#include "STime.hh"
#include "SEvt.hh"

#ifdef WITH_PMTSIM
#include "J_PMTSIM_LOG.hh"
#include "PMTSim.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
#ifdef WITH_PMTSIM
    J_PMTSIM_LOG_(0); 
#endif
    int VERSION = ssys::getenvint("VERSION", 0 );  
    LOG(info) << "[ " << argv[0] << " " << STime::Now() << " VERSION " << VERSION ; 

    LOG(info) << U4Recorder::Switches() ; 

    U4App* app = U4App::Create() ;  
    app->BeamOn(); 
    delete app ; 

    LOG(info) << "] " << argv[0] << " " << STime::Now() << " VERSION " << VERSION << " savedir " << SEvt::GetSaveDir() ; 
    return 0 ; 
}

