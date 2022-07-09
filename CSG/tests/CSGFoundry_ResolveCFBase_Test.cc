/**
CSGFoundry_ResolveCFBase_Test.cc
==================================


::

    epsilon:tests blyth$ GEOM=hello hello_CFBaseFromGEOM=/red/green/blue CSGFoundry_ResolveCFBase_Test
    2022-07-09 18:42:27.049 FATAL [4021970] [*CSGFoundry::ResolveCFBase@2381]  cfbase/CSGFoundy directory /red/green/blue/CSGFoundry IS NOT READABLE 
    -
    epsilon:tests blyth$ GEOM=hello hello_CFBaseFromGEOM=/tmp  CSGFoundry_ResolveCFBase_Test
    2022-07-09 18:42:44.516 FATAL [4022121] [*CSGFoundry::ResolveCFBase@2381]  cfbase/CSGFoundy directory /tmp/CSGFoundry IS NOT READABLE 
    -
    epsilon:tests blyth$ mkdir /tmp/CSGFoundry
    epsilon:tests blyth$ GEOM=hello hello_CFBaseFromGEOM=/tmp  CSGFoundry_ResolveCFBase_Test
    /tmp
    epsilon:tests blyth$ 


**/

#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = CSGFoundry::ResolveCFBase(); 
    std::cout << ( cfbase ? cfbase : "-" ) << std::endl ;  

    return 0 ; 
}

