
#include "SSys.hh"

#include "NScene.hpp"
#include "NSceneConfig.hpp"
#include "NPY.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    // $TMP/tgltf-t-/sc.gltf
    const char* gltfbase = argc > 1 ? argv[1] : "$TMP/tgltf-t-" ;
    const char* gltfname = "sc.gltf" ;
    const char* gltfconfig = "check_surf_containment=1,check_aabb_containment=0" ; 




    if(!NScene::Exists(gltfbase, gltfname))

    {
        LOG(warning) << "no such scene at"
                     << " base " << gltfbase
                     << " name " << gltfname
                     ;
        return 0 ; 
    } 
   

    int dbgnode = SSys::getenvint("DBGNODE", -1) ; 

    const char* idfold = NULL ; 
    NSceneConfig* config = new NSceneConfig(gltfconfig);
    NScene* scene = NScene::Load( gltfbase, gltfname, idfold, config, dbgnode );

    assert(scene);
    
    //scene->dumpNdTree();


    return 0 ; 
}
