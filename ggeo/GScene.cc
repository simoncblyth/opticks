#include "OpticksConst.hh"

#include "NScene.hpp"
#include "NTrianglesNPY.hpp"
#include "NSensor.hpp"
#include "NCSG.hpp"
#include "Nd.hpp"
#include "NGLMExt.hpp"
#include "NPY.hpp"


#include "GGeo.hh"
#include "GParts.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMatrix.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GScene.hh"
#include "GMergedMesh.hh"
#include "PLOG.hh"

GScene::GScene( GGeo* ggeo, NScene* scene )
    :
    m_ggeo(ggeo),
    m_geolib(ggeo->getGeoLib()),
    m_bndlib(ggeo->getBndLib()),
    m_scene(scene),
    m_root(NULL)
{
    init();
}

void GScene::init()
{
    modifyGeometry();

    importMeshes(m_scene);

    m_root = createVolumeTree(m_scene) ;

    createInstancedMergedMeshes(false);
}


void GScene::modifyGeometry()
{
    // is this needed ?? 
    // the merged meshes are created via GGeoLib which
    // holds onto them 

    m_geolib->clear();
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
    LOG(debug) << "GScene::createVolumeTree START" ; 
    assert(scene);

    //scene->dumpNdTree("GScene::createVolumeTree");

    GSolid* root = createVolumeTree_r( scene->getRoot() );
    assert(root);

    LOG(info) << "GScene::createVolumeTree DONE num_nodes: " << m_nodes.size()  ; 


    return root ; 
}

GSolid* GScene::createVolumeTree_r(nd* n)
{
    GSolid* node = createVolume(n);

    // First try simple strategy of instancing every mesh 
    // this differs from the subtree instancing of GTreeCheck.
    // Applying instancing to every distinct mesh
    // will result in OpenGL draw calls for each distinct mesh (~250)
    // compared to subtree instancing that yields ~5 draw calls.
    // Nevetherless the simplicity of having every mesh instanced 
    // makes it interesting to see how it performs.
    //
    // With this simple strategy just the target node will get merged in
    // GMergedMesh::traverse_r

    unsigned mesh_idx = n->mesh ;   
    node->setRepeatIndex(mesh_idx);  // <-- steers GMergedMesh::create repsel


    typedef std::vector<nd*> VN ; 
    for(VN::const_iterator it=n->children.begin() ; it != n->children.end() ; it++)
    {
        nd* cn = *it ; 
        GSolid* child = createVolumeTree_r(cn);
        node->addChild(child);
    } 


    unsigned node_idx = n->idx ;
    assert(m_nodes.count(node_idx) == 0); 
    m_nodes[node_idx] = node ; 

    return node  ; 
}


GSolid* GScene::getNode(unsigned node_idx)
{
    assert(node_idx < m_nodes.size());
    return m_nodes[node_idx];  
}


GSolid* GScene::createVolume(nd* n)
{
    assert(n);
    // TODO: avoid duplication between this and GMaker

    unsigned node_idx = n->idx ;
    unsigned mesh_idx = n->mesh ; 
    std::string bnd = n->boundary ; 

    const char* spec = bnd.c_str();

    LOG(debug) << "GScene::createVolume"
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

    pts->setBndLib(m_bndlib);

    solid->setParts( pts );

    return solid ; 
}



void GScene::createInstancedMergedMeshes(bool /*delta*/)
{
    LOG(info) << "GScene::createInstancedMergedMeshes START " ; 
    makeMergedMeshAndInstancedBuffers() ; 
    LOG(info) << "GScene::createInstancedMergedMeshes DONE" ; 
}



void GScene::makeMergedMeshAndInstancedBuffers()
{
    unsigned num_meshes = m_scene->getNumMeshes();
    for(unsigned mesh_idx=0 ; mesh_idx < num_meshes ; mesh_idx++)  // 1-based index
    {
         const std::vector<unsigned>& instances = m_scene->getInstances(mesh_idx);
         assert(instances.size() > 0 ); 

         unsigned first_node_idx = instances[0]; 
         unsigned ridx = mesh_idx ; 

         GSolid* rbase = getNode(first_node_idx);
         assert(rbase);

         GMergedMesh* mm = m_ggeo->makeMergedMesh(ridx, rbase); 
         makeInstancedBuffers(mm, ridx);

         mm->setGeoCode(OpticksConst::GEOCODE_ANALYTIC);


         GParts* combi = GParts::combine( rbase->getParts()  );

         mm->setParts( combi ) ;  // combine even when only 1 for consistent handling 

    }
}

void GScene::makeInstancedBuffers(GMergedMesh* mm, unsigned int ridx)
{
    LOG(info) << "GScene::makeInstancedBuffers" << " ridx " << ridx ;
     //mm->dumpSolids("GTreeCheck::makeInstancedBuffers dumpSolids");

     NPY<float>* itr = makeInstanceTransformsBuffer(ridx); 
     mm->setITransformsBuffer(itr);

     NPY<unsigned>* iid = makeInstanceIdentityBuffer(ridx);
     mm->setInstancedIdentityBuffer(iid);

     NPY<unsigned>* aii = makeAnalyticInstanceIdentityBuffer(ridx);
     mm->setAnalyticInstancedIdentityBuffer(aii);
}

NPY<float>* GScene::makeInstanceTransformsBuffer(unsigned ridx)
{
    unsigned mesh_idx = ridx ; // simple 1-to-1 mesh to instance
    return m_scene->makeInstanceTransformsBuffer(mesh_idx)  ; 
}

NPY<unsigned>* GScene::makeInstanceIdentityBuffer(unsigned ridx) 
{
    unsigned mesh_idx = ridx ; // simple 1-to-1 mesh to instance
    const std::vector<unsigned>& instances = m_scene->getInstances(mesh_idx);
    unsigned num_instances = instances.size() ; 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 4);
    buf->zero(); 

    for(unsigned i=0 ; i < num_instances ; i++)
    {
        unsigned node_idx = instances[i];
        GSolid* node = getNode(node_idx);
        assert(node) ; 

        guint4 id = node->getIdentity();
        glm::uvec4 uid(id.x, id.y, id.z, id.w);
        buf->setQuadU( uid, i );
    }
    return buf ;  
}

NPY<unsigned>* GScene::makeAnalyticInstanceIdentityBuffer(unsigned ridx) 
{
    unsigned mesh_idx = ridx ; // simple 1-to-1 mesh to instance
    const std::vector<unsigned>& instances = m_scene->getInstances(mesh_idx);
    unsigned num_instances = instances.size() ; 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 1, 4);  //  TODO: unify shape aii and ii shape
    buf->zero(); 

    for(unsigned int i=0 ; i < num_instances ; i++) // over instances of the same geometry
    {
        unsigned node_idx = instances[i];
        GSolid* node = getNode(node_idx);
        assert(node) ; 
      
        NSensor* ss = node->getSensor();
        unsigned int sid = ss && ss->isCathode() ? ss->getId() : 0 ;

        if(sid > 0)
            LOG(debug) << "GScene::makeAnalyticInstanceIdentityBuffer " 
                      << " sid " << std::setw(10) << std::hex << sid << std::dec 
                      << " ss " << (ss ? ss->description() : "NULL" )
                      ;

        glm::uvec4 aii ; 
        aii.x = node_idx ;        
        aii.y = i ;                      // instance index (for triangulated this contains the mesh index)
        aii.z = 0 ;                      // formerly boundary, but with analytic have broken 1-1 solid/boundary relationship so boundary must live in partBuffer
        aii.w = NSensor::RefIndex(ss) ;  // the only critical one 

        buf->setQuadU(aii, i, 0); 
        
    }
    return buf ; 
}





/*

385 void GMergedMesh::mergeSolid( GSolid* solid, bool selected )
386 {
387     GMesh* mesh = solid->getMesh();
388     unsigned int nvert = mesh->getNumVertices();
389     unsigned int nface = mesh->getNumFaces();
390     guint4 _identity = solid->getIdentity();
391 
392     GNode* base = getCurrentBase();
393     GMatrixF* transform = base ? solid->getRelativeTransform(base) : solid->getTransform() ;

        // after base global, or totally global transform persisted in the node

394     gfloat3* vertices = mesh->getTransformedVertices(*transform) ;

*/








