
#include "OPTICKS_LOG.hh"
#include "YOGMaker.hh"
#include "YOGGeometry.hh"

using YOG::Geometry ; 
using YOG::Maker ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    Geometry geom(3) ; 
    geom.make_triangle();

    Maker ym ; 
    ym.demo_create(geom); 

    const char* path = "/tmp/YOGMakerTest/YOGMakerTest.gltf" ; 
    ym.save(path);


    return 0 ; 
}


