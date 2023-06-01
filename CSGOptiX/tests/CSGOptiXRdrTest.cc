/**
CSGOptiXRdrTest : minimal variant of CSGOptiXRenderTest for debugging single renders
======================================================================================


**/

#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SSim.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SEventConfig::SetRGMode("render"); 
    SSim::Create(); 
    
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;  // uploads fd and then instanciates 
    cx->render_snap(); 

    return 0 ;
}



