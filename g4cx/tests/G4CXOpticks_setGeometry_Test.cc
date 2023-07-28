/**
G4CXOpticks_setGeometry_Test.cc
=================================

Action depends on envvars such as OpticksGDMLPath, see G4CXOpticks::setGeometry

**/

#include "OPTICKS_LOG.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4CXOpticks::SetGeometry();  

    return 0 ; 
}
