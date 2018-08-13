#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "OpMgr.hh"

/**
OpSnapTest (formerly OpTest)
=============================

Loads geometry from cache, creates sequence of ppm raytrace snapshots of geometry::

    OpSnapTest 
        triangulated geometry default

    OpSnapTest --gltf 3
        FAILS : the old default geocache has some issues 

    OPTICKS_RESOURCE_LAYOUT=104 OpSnapTest --gltf 3
        succeeds with the ab- validated geocache : creating analytic raytrace snapshots

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    Opticks ok(argc, argv, "--tracer"); 
    OpMgr op(&ok);
    op.snap();
    return 0 ; 
}





