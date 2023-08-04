/**
G4CXAppTest.cc
===============

Starting from U4App.h/U4SimulateTest.cc create G4CXApp.h/G4CXAppTest.cc 

Aim is to incorporate G4CXOpticks into the U4Recorder enabled Geant4 application
to provide standalone bi-simulation. 

**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "G4CXApp.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4CXApp* app = G4CXApp::Create() ;   
    app->BeamOn(); 
    delete app ;  // avoids "Attempt to delete the (physical volume/logical volume/solid/region) store while geometry closed" warnings 
     
    return 0 ; 
}


