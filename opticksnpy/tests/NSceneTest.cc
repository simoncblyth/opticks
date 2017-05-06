#include "NScene.hpp"
#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* base = argc > 1 ? argv[1] : "$TMP/nd" ;
    const char* name = "scene.gltf" ;

    NScene* scene = NScene::load( base, name  ); 
    assert(scene);

    return 0 ; 
}
