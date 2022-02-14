/**
CSGIntersectSolidTest
=======================

Used from script CSG/csg_geochain.sh 

Simple single solid testing of CSG intersects obtained using 
code intended for GPU but running on CPU in order to facilitate 
convenient debugging.

Initially thought to model on extg4/tests/X4IntersectSolidTest 
but actually best to follow cx cxs_geochain.sh (not x4 xxs.sh) 
as access to geochain or CSGMaker geometry is then the same 

**/

#include "OPTICKS_LOG.hh"
#include "CSGGeometry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGGeometry geom ;
    geom() ; 

    return 0 ; 

}




