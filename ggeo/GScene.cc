#include "NScene.hpp"
#include "GGeo.hh"
#include "GScene.hh"
#include "PLOG.hh"


GScene::GScene( GGeo* ggeo, NScene* scene )
    :
    m_ggeo(ggeo),
    m_scene(scene)
{
    init();
}


void GScene::init()
{
    createInstancedMergedMeshes(false);
}

unsigned GScene::getNumRepeats()
{
    return 0 ; 
}

void GScene::createInstancedMergedMeshes(bool /*delta*/)
{
    LOG(info) << "GScene::createInstancedMergedMeshes" ; 

    // GMergedMesh::traverse needs a GNode tree 
    //  ... need to revivify the GNode/GSolid tree from the NScene ? 

}




NPY<float>* GScene::makeInstanceTransformsBuffer(unsigned int ridx)
{
    LOG(info) << "GScene::makeInstanceTransformsBuffer"
              << " ridx " << ridx 
              ;
    return NULL ; 
}
 

