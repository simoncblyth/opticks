#include "NScene.hpp"
#include "NTrianglesNPY.hpp"
#include "NCSG.hpp"
#include "Nd.hpp"
#include "NGLMExt.hpp"


#include "GGeo.hh"
#include "GParts.hh"
#include "GBndLib.hh"
#include "GMatrix.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GScene.hh"
#include "PLOG.hh"

GScene::GScene( GGeo* ggeo, NScene* scene )
    :
    m_ggeo(ggeo),
    m_bndlib(ggeo->getBndLib()),
    m_scene(scene),
    m_root(NULL)
{
    init();
}

void GScene::init()
{
    importMeshes(m_scene);
    m_root = createVolumeTree(m_scene) ;
    createInstancedMergedMeshes(false);
}

void GScene::importMeshes(NScene* scene)
{
    unsigned num_meshes = scene->getNumMeshes();
    LOG(info) << "GScene::importMeshes num_meshes " << num_meshes  ; 
     
    for(unsigned mesh_idx=0 ; mesh_idx < num_meshes ; mesh_idx++)
    {
        NCSG* csg = scene->getCSG(mesh_idx);
        NTrianglesNPY* tris = csg->getTris();
        assert(tris);
        assert( csg->getIndex() == mesh_idx) ;
        GMesh* mesh = GMesh::make_mesh(tris->getTris(), mesh_idx );
        m_meshes[mesh_idx] = mesh ; 
        assert(mesh);
    }
}

GMesh* GScene::getMesh(unsigned mesh_idx)
{
    assert( mesh_idx < m_meshes.size() );
    return m_meshes[mesh_idx];
}
NCSG* GScene::getCSG(unsigned mesh_idx)
{
    return m_scene->getCSG(mesh_idx);
}











GSolid* GScene::createVolumeTree(NScene* scene)
{
    LOG(info) << "GScene::createVolumeTree" ; 
    assert(scene);

    scene->dumpNdTree("GScene::createVolumeTree");

    GSolid* root = createVolumeTree_r( scene->getRoot() );
    assert(root);
    return root ; 
}

GSolid* GScene::createVolumeTree_r(nd* n)
{
    GSolid* node = createVolume(n);
    typedef std::vector<nd*> VN ; 
    for(VN::const_iterator it=n->children.begin() ; it != n->children.end() ; it++)
    {
        nd* cn = *it ; 
        GSolid* child = createVolumeTree_r(cn);
        node->addChild(child);
    } 
    return node  ; 
}


GSolid* GScene::createVolume(nd* n)
{
    assert(n);
    // TODO: avoid duplication between this and GMaker

    unsigned node_idx = n->idx ;
    unsigned mesh_idx = n->mesh ; 
    std::string bnd = n->boundary ; 

    const char* spec = bnd.c_str();

    LOG(info) << "GScene::createVolume"
              << " node_idx " << std::setw(5) << node_idx 
              << " mesh_idx " << std::setw(3) << mesh_idx 
              << " bnd " << bnd 
              ;

    assert(!bnd.empty());

    GMesh* mesh = getMesh(mesh_idx);

    NCSG* csg = getCSG(mesh_idx);

    glm::mat4 xf_local = n->transform->t ;    

    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(xf_local));

    GSolid* solid = new GSolid(node_idx, transform, mesh, UINT_MAX, NULL );     

    solid->setSensor( NULL );      

    solid->setCSGFlag( csg->getRootType() );
  
    unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing

    solid->setBoundary(boundary);     // unlike ctor these create arrays

    GParts* pts = GParts::make( csg, spec  );

    solid->setParts( pts );

    return solid ; 
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
 

