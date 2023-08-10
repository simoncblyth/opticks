/**
G4CXTest.cc : Standalone bi-simulation
==========================================

This was based on the Geant4 application u4/tests/U4SimulateTest.cc (U4App.h)
with the addition of only a few lines to incorporate the G4CXOpticks GPU simulation. 

**/

#include "OPTICKS_LOG.hh"
#include "G4CXApp.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    G4CXApp::Main(); 
    return 0 ; 
}

