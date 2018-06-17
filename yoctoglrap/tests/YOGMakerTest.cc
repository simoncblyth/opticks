
#include "OPTICKS_LOG.hh"
#include "BFile.hh"

#include "YOGMaker.hh"
#include "YOG.hh"
#include "YOGGeometry.hh"

using YOG::Geometry ; 
using YOG::Sc ; 
using YOG::Maker ; 

void test_demo_geom()
{
    Geometry geom(3) ; 
    geom.make_triangle();

    Sc* sc = new Sc(0) ;  
    Maker ym(sc) ; 
    ym.demo_create(geom); 

    std::string path = BFile::FormPath("$TMP/yoctoglrap/tests/YOGMakerTest/YOGMakerTest.gltf");

    ym.convert();
    ym.save(path.c_str());
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_demo_geom();

    return 0 ; 
}


