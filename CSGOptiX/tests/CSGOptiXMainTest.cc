/**
CSGOptiXMainTest
==================

This "proceed" approach means that a single executable 
does very different things depending on the RGMode envvar. 
That is not convenient as default bookkeeping is based on 
executable names so instead use three separate executables 
that each use the corresponding SimulateMain, RenderMain, SimtraceMain 
static method. 

**/

#include "OPTICKS_LOG.hh"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    CSGOptiX::Main(); 
    return 0 ;
}



