
#include "SSys.hh"
#include "BOpticksResource.hh"


#include "NScene.hpp"
#include "NPY.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 


    BOpticksResource okr ;  // no Opticks at this level 


    const char* dbgmesh = SSys::getenvvar("DBGMESH");
    int dbgnode = SSys::getenvint("DBGNODE", -1) ; 

    const char* gltfbase = argc > 1 ? argv[1] : okr.getDebuggingIDFOLD() ;
    const char* gltfname = "g4_00.gltf" ;
    const char* gltfconfig = "check_surf_containment=0,check_aabb_containment=0" ; 

    LOG(info) << argv[0]
              << " gltfbase " << gltfbase
              << " gltfname " << gltfname
              << " gltfconfig " << gltfconfig
              ;


    if(!NScene::Exists(gltfbase, gltfname))
    {
        LOG(warning) << "no such scene at"
                     << " base " << gltfbase
                     << " name " << gltfname
                     ;
        return 0 ; 
    } 

    NScene* scene = NScene::Load( gltfbase, gltfname, gltfconfig, dbgnode );
    assert(scene);

    scene->dumpCSG(dbgmesh); 


    return 0 ; 
}
