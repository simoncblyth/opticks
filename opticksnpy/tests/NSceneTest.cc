#include "NScene.hpp"
#include "NSceneConfig.hpp"
#include "NPY.hpp"
#include "NPY_LOG.hh"

#include "PLOG.hh"


void test_makeInstanceTransformsBuffer(NScene* scene)
{
    unsigned num_mesh = scene->getNumMeshes();
    for(unsigned mesh_idx=0 ; mesh_idx < num_mesh ; mesh_idx++)
    {    
        NPY<float>* buf = scene->makeInstanceTransformsBuffer(mesh_idx);
        std::cout 
           << " mesh_idx " << std::setw(3) << mesh_idx 
           << " instance transforms " << std::setw(3) << buf->getNumItems()
           << std::endl ;  
        buf->dump();
        scene->dumpAllInstances(mesh_idx); 
    }
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* base = argc > 1 ? argv[1] : "$TMP/nd" ;
    const char* name = "scene.gltf" ;
    const char* config = NULL ; 

    if(!NScene::Exists(base, name))
    {
        LOG(warning) << "no such scene at"
                     << " base " << base
                     << " name " << name
                     ;
        return 0 ; 
    } 


    int dbgnode = -1 ; 
    int scene_idx = 0 ; 


    NSceneConfig* config_ = new NSceneConfig(config);
    NScene* scene = new NScene( base, name, config_, dbgnode, scene_idx  ); 
    assert(scene);
    scene->dump(argv[0]);
    scene->dumpNdTree();

    //scene->dumpAll();

    test_makeInstanceTransformsBuffer(scene);

    return 0 ; 
}
