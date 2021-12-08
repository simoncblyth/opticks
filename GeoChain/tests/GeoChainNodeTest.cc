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

    std::stringstream ss ; 
    ss << "creator:GeoChainNodeTest" << std::endl ; 
    ss << "name:" << name << std::endl ; 
    std::string meta = ss.str(); 
    LOG(info) << meta ; 

    nnode* root = nullptr ; 
    if(strcmp(name, "sphere") == 0)
    {
        root = make_sphere( 0.f, 0.f, 0.f, 100.f ); 
    }

    assert( root ); 

    const char* base = GeoChain::BASE ; 

    unsetenv("OPTICKS_KEY"); 
    const char* argforced = "--allownokey --gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    GeoChain chain(&ok); 
    chain.convertNodeTree(root, meta);  
    chain.save(base, name); 

    return 0 ; 
}

