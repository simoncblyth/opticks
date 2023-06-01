/**
CSGOptiXRdrTest : more minimal version of CSGOptiXRenderTest
==============================================================


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
 




    return 0 ;
}



