/**
CSGGeometryTest
=======================

**/

#include "OPTICKS_LOG.hh"
#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGGeometry geom ;
    geom.dump(); 
    std::cout << geom.desc() << std::endl ; 

    return 0 ; 

}




