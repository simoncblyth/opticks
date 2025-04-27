/**
G4CXTest.cc : Standalone bi-simulation
======================================

**/

#include "OPTICKS_LOG.hh"
#include "G4CXApp.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    return G4CXApp::Main();
}

