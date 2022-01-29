/**
CSGGeometryTest
=======================

**/

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGGeometry geom ;
    geom.dump(); 
    geom.draw(); 

    return 0 ; 

}




