/**
GeoChainNodeTest.cc : testing the chain of geometry conversions for NNode trees
==========================================================================================


**/

#include <cassert>
#include <cstdlib>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "GeoChain.hh"

#include "NSphere.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* name = SSys::getenvvar("GEOM", "sphere" ); 
    nnode* root = nullptr ; 
    if(strcmp(name, "sphere") == 0)
    {
        root = nsphere::Create( 0.f, 0.f, 0.f, 100.f ); 
    }

    assert( root ); 


    unsetenv("OPTICKS_KEY"); 
    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    GeoChain gc(&ok); 
    gc.convertNodeTree(root);  
    gc.save(name); 

    return 0 ; 
}

