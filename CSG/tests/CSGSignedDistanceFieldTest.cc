#include "CSGGeometry.h"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGGeometry geom;
    geom.saveSignedDistanceField(); 

    return 0 ; 
}
