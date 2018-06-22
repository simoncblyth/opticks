
#include "SSys.hh"
#include "BFile.hh"

#include "Nd.hpp"
#include "NGLTF.hpp"
#include "NScene.hpp"
#include "NSceneConfig.hpp"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    const char* default_path = "$TMP/tgltf-t-/sc.gltf" ; 
    //const char* default_path = "/tmp/g4_00.gltf" ; 
    const char* path = argc > 1 ? argv[1] : default_path  ; 

    std::string base = BFile::ParentDir(path);
    std::string name = BFile::Name(path);
  
    //const char* gltfconfig = "check_surf_containment=1,check_aabb_containment=0" ; 
    const char* gltfconfig = "check_surf_containment=0,check_aabb_containment=0" ; 

    if(!NGLTF::Exists(base.c_str(), name.c_str()))
    {
        LOG(warning) << "no such scene at"
                     << " base " << base
                     << " name " << name
                     ;
        return 0 ; 
    } 
   
    int dbgnode = SSys::getenvint("DBGNODE", -1) ; 
    const char* idfold = NULL ; 
    NSceneConfig* config = new NSceneConfig(gltfconfig);
    NScene* scene = NScene::Load( base.c_str(), name.c_str(), idfold, config, dbgnode );

    assert(scene);

    assert( scene->getRoot() == nd::get(0) );

    LOG(info) << " nd::num_nodes() " << nd::num_nodes()  ; 

    
    //scene->dumpNdTree();


    return 0 ; 
}
