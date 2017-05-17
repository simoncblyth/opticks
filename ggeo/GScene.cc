#include "OpticksConst.hh"

#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NScene.hpp"
#include "NSensor.hpp"
#include "NCSG.hpp"
#include "Nd.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"


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
    dumpMeshes();

    m_root = createVolumeTree(m_scene) ;


    createInstancedMergedMeshes(true);
    dumpMergedMeshes();
}


void GScene::modifyGeometry()
{
    // clear out the G4DAE geometry GMergedMesh, typically loaded from cache 
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
        // maybe GGeo should be holding on to these ?
        assert(mesh);
    }
}


unsigned GScene::getNumMeshes()
{
   return m_meshes.size();
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

void GScene::dumpMeshes()
{
    unsigned num_meshes = getNumMeshes() ; 
    LOG(info) << "GScene::dumpMeshes" 
              << " num_meshes " << num_meshes 
              ;

    for(unsigned mesh_idx=0 ; mesh_idx < num_meshes ; mesh_idx++)
    {
         GMesh* mesh = getMesh( mesh_idx );
         gbbox bb = mesh->getBBox(0) ; 

         std::cout << std::setw(3) << mesh_idx 
                   << " "
                   << bb.description()
                   << std::endl ; 
    }
}


GSolid* GScene::createVolumeTree(NScene* scene)
{
    LOG(info) << "GScene::createVolumeTree START" ; 
    assert(scene);

    //scene->dumpNdTree("GScene::createVolumeTree");

    nd* root_nd = scene->getRoot() ;
    assert(root_nd->idx == 0 );


    GSolid* root = createVolumeTree_r( root_nd, NULL );
    assert(root);

    LOG(info) << "GScene::createVolumeTree DONE num_nodes: " << m_nodes.size()  ; 
    return root ; 
}


GSolid* GScene::createVolumeTree_r(nd* n, GSolid* parent)
{
    GSolid* node = createVolume(n);
    node->setParent(parent) ;   // tree hookup 

    typedef std::vector<nd*> VN ; 
    for(VN::const_iterator it=n->children.begin() ; it != n->children.end() ; it++)
    {
        nd* cn = *it ; 
        GSolid* child = createVolumeTree_r(cn, node);
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

    unsigned node_idx = n->idx ;
    unsigned mesh_idx = n->mesh ; 

    int ridx = n->repeatIdx ; 

    std::string bnd = n->boundary ; 

    const char* spec = bnd.c_str();

    LOG(info) << "GScene::createVolume"
              << " node_idx " << std::setw(5) << node_idx 
              << " mesh_idx " << std::setw(3) << mesh_idx 
              << " ridx " << std::setw(3) << ridx 
              << " bnd " << bnd 
              ;

    assert(!bnd.empty());
    assert(ridx > -1);

    GMesh* mesh = getMesh(mesh_idx);

    NCSG* csg = getCSG(mesh_idx);


    glm::mat4 xf_global = n->gtransform->t ;    

    glm::mat4 xf_local  = n->transform->t ;    

    GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));

    GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));


    GSolid* solid = new GSolid(node_idx, gtransform, mesh, UINT_MAX, NULL );     

    solid->setLevelTransform(ltransform); 

    // see AssimpGGeo::convertStructureVisit

    solid->setSensor( NULL );      

    solid->setCSGFlag( csg->getRootType() );
  
    unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing

    solid->setBoundary(boundary);     // unlike ctor these create arrays


    GParts* pts = GParts::make( csg, spec  ); // amplification from mesh level to node level 

    pts->setBndLib(m_bndlib);

    solid->setParts( pts );

    solid->setRepeatIndex( n->repeatIdx ); 


    return solid ; 
}




void GScene::deltacheck()
{
    // check consistency of the level transforms
    assert(m_root);
    deltacheck_r(m_root, 0);
}

void GScene::deltacheck_r( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GMatrixF* gtransform = solid->getTransform();

    //GMatrixF* ltransform = solid->getLevelTransform();  
    GMatrixF* ctransform = solid->calculateTransform();
    float delta = gtransform->largestDiff(*ctransform);

    std::cout << "GScene::deltacheck_r gtransform " << gtransform->brief(7) << std::endl  ;

    assert(delta < 1e-6) ;

    for(unsigned int i = 0; i < node->getNumChildren(); i++) deltacheck_r(node->getChild(i), depth + 1 );
}




void GScene::createInstancedMergedMeshes(bool delta)
{
    
    if(delta)
    {  
        deltacheck();
    }

    makeMergedMeshAndInstancedBuffers() ; 
}



void GScene::makeMergedMeshAndInstancedBuffers()
{
    
    unsigned num_repeats = m_scene->getNumRepeats(); // global 0 included
    unsigned nmm_created = 0 ; 

    for(unsigned ridx=0 ; ridx < num_repeats ; ridx++)
    {
         LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers ridx " << ridx << " START " ;  

         bool inside = ridx == 0 ? true : false ; 
         const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside );
         if(instances.size() == 0)
         {
             LOG(warning) << "GScene::makeMergedMeshAndInstancedBuffers"
                          << " no instances with ridx " << ridx
                          ;
             continue ; 
         } 

         GSolid* instance0 = dynamic_cast<GSolid*>(instances[0]); 

         GSolid* base = ridx == 0 ? NULL : instance0 ; 

         GMergedMesh* mm = m_ggeo->makeMergedMesh(ridx, base, m_root );   // TODO: check off-by-1 in base transforms

         assert(mm);

         makeInstancedBuffers(mm, ridx);

         mm->setGeoCode(OpticksConst::GEOCODE_ANALYTIC);

         std::cout << "GScene::makeMergedMeshAndInstancedBuffers"
                   << " ridx " << ridx 
                   << " mm " << mm->getIndex()
                   << " nmm_created " << nmm_created
                   << std::endl ; 

         nmm_created++ ; 

         GMergedMesh* mmc = m_ggeo->getMergedMesh(ridx);
         assert(mmc == mm);

         LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers ridx " << ridx << " DONE " ;  
    }

    unsigned nmm = m_ggeo->getNumMergedMesh();
   
     
    LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers"
              << " num_repeats " << num_repeats
              << " nmm_created " << nmm_created
              << " nmm " << nmm
               ; 

    assert(nmm == nmm_created);


}


void GScene::dumpMergedMeshes()
{
    int nmm = m_ggeo->getNumMergedMesh();
    int mia = 0 ;

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        std::cout << std::setw(4) << i << " mm? " << (mm ? int(mm->getIndex()) : -1 ) << std::endl ; 
        if(mm == NULL) mia++ ; 
    } 

    LOG(info) << "GScene::dumpMergedMeshes" 
              << " nmm " << nmm
              << " mia " << mia
              ;

    assert(mia == 0 );
}



void GScene::makeInstancedBuffers(GMergedMesh* mm, unsigned ridx)
{
    bool inside = ridx == 0 ; 
    const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside );
    unsigned num_instances = instances.size(); 

    LOG(info) << "GScene::makeInstancedBuffers" 
              << " ridx " << std::setw(3) << ridx
              << " num_instances " << std::setw(5) << num_instances
              ;

    NPY<float>* itr = makeInstanceTransformsBuffer(instances, ridx); 
    mm->setITransformsBuffer(itr);

    NPY<unsigned>* iid = makeInstanceIdentityBuffer(instances, ridx);
    mm->setInstancedIdentityBuffer(iid);

    NPY<unsigned>* aii = makeAnalyticInstanceIdentityBuffer(instances, ridx);
    mm->setAnalyticInstancedIdentityBuffer(aii);
}


NPY<float>* GScene::makeInstanceTransformsBuffer(const std::vector<GNode*>& instances, unsigned ridx)
{
    NPY<float>* buf = NULL ; 
    if(ridx == 0)
    {
        buf = NPY<float>::make_identity_transforms(1) ; 
    }
    else
    {
        unsigned num_instances = instances.size(); 
        buf = NPY<float>::make(num_instances, 4, 4);
        buf->zero(); 
        for(unsigned i=0 ; i < num_instances ; i++)
        {
            GNode* instance = instances[i] ;
            GMatrix<float>* gtransform = instance->getTransform();
            const float* data = static_cast<float*>(gtransform->getPointer());
            glm::mat4 xf_global = glm::make_mat4( data ) ;  
            buf->setMat4(xf_global, i);  
        } 
    }
    return buf ; 
}

NPY<unsigned>* GScene::makeInstanceIdentityBuffer(const std::vector<GNode*>& instances, unsigned /*ridx*/)   
{
    unsigned num_instances = instances.size(); 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 4);
    buf->zero(); 
    for(unsigned i=0 ; i < num_instances ; i++)
    {
        GSolid* instance = dynamic_cast<GSolid*>(instances[i]) ; 
        guint4 id = instance->getIdentity();
        glm::uvec4 uid(id.x, id.y, id.z, id.w);
        buf->setQuadU( uid, i );
    }
    return buf ;  
}

NPY<unsigned>* GScene::makeAnalyticInstanceIdentityBuffer( const std::vector<GNode*>& instances, unsigned /*ridx*/) 
{
    unsigned num_instances = instances.size(); 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 1, 4);  //  TODO: unify shape aii and ii shape
    buf->zero(); 

    for(unsigned int i=0 ; i < num_instances ; i++) // over instances of the same geometry
    {
        GSolid* instance = dynamic_cast<GSolid*>(instances[i]) ; 
        NSensor* ss = instance->getSensor();
        unsigned int sid = ss && ss->isCathode() ? ss->getId() : 0 ;

        if(sid > 0)
            LOG(debug) << "GScene::makeAnalyticInstanceIdentityBuffer " 
                      << " sid " << std::setw(10) << std::hex << sid << std::dec 
                      << " ss " << (ss ? ss->description() : "NULL" )
                      ;

        glm::uvec4 aii ; 
        aii.x = instance->getIndex()  ;        
        aii.y = i ;                      // instance index (for triangulated this contains the mesh index)
        aii.z = 0 ;                      // formerly boundary, but with analytic have broken 1-1 solid/boundary relationship so boundary must live in partBuffer
        aii.w = NSensor::RefIndex(ss) ;  // the only critical one 

        buf->setQuadU(aii, i, 0); 
        
    }
    return buf ; 
}


