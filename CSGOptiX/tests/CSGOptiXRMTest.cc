/**
CSGOptiXRMTest : minimal variant of CSGOptiXRenderTest for debugging single renders
======================================================================================

**/

#include "OPTICKS_LOG.hh"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    CSGOptiX::RenderMain(); 
    return 0 ;
}



