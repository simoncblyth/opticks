/**

GSceneTest --okcore debug --gltfname hello.gltf

**/
#include <set>
#include <string>

#include "NScene.hpp"

#include "Opticks.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "GScene.hh"

#include "PLOG.hh"
#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG__ ;
    NPY_LOG__ ;
    GGEO_LOG__ ;

    Opticks ok(argc, argv, "--gltf 101");
    ok.configure();

    const char* base = ok.getGLTFBase() ;
    const char* name = ok.getGLTFName() ;
    const char* config = ok.getGLTFConfig() ;
    int gltf = ok.getGLTF();

    assert(gltf == 101);

    LOG(info) << argv[0]
              << " base " << base
              << " name " << name
              << " config " << config
              << " gltf " << gltf 
              ; 

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.loadFromGLTF();
    gg.dumpStats();



    return 0 ; 
}


