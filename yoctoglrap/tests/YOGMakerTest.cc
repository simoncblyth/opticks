
#include "OPTICKS_LOG.hh"
#include "YOGMaker.hh"
#include "YOGGeometry.hh"

using YOG::Geometry ; 
using YOG::Maker ; 

void test_demo_geom()
{
    Geometry geom(3) ; 
    geom.make_triangle();

    Maker ym(NULL) ; 
    ym.demo_create(geom); 

    const char* path = "/tmp/YOGMakerTest/YOGMakerTest.gltf" ; 

    ym.convert();
    ym.save(path);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    test_demo_geom();

    return 0 ; 
}


