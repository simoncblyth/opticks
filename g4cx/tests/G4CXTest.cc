/**
G4CXAppTest.cc
===============

Starting from U4App.h/U4SimulateTest.cc create G4CXApp.h/G4CXAppTest.cc 

Aim is to incorporate G4CXOpticks into the U4Recorder enabled Geant4 application
to provide standalone bi-simulation. 

**/

#include "OPTICKS_LOG.hh"
#include "G4CXApp.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    G4CXApp::Main(); 
    return 0 ; 
}


