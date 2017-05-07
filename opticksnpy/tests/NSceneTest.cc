#include "NScene.hpp"
#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* base = argc > 1 ? argv[1] : "$TMP/nd" ;
    const char* name = "scene.gltf" ;

    NScene* scene = new NScene( base, name  ); 
    assert(scene);
    scene->dump(argv[0]);

    //scene->dumpAll();



    return 0 ; 
}
