#include "OpticksConst.hh"

#include "NGLM.hpp"
#include "NGLMExt.hpp"

#include "NScene.hpp"

#include "NSensor.hpp"
#include "NCSG.hpp"
#include "Nd.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NSensorList.hpp"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GItemIndex.hh"
#include "GGeo.hh"
#include "GParts.hh"
#include "GGeoLib.hh"
#include "GNodeLib.hh"
#include "GBndLib.hh"
#include "GMatrix.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GScene.hh"
#include "GMergedMesh.hh"


#include "PLOG.hh"

GScene::GScene( Opticks* ok, GGeo* ggeo )
    :
    m_ok(ok),
    m_gltf(m_ok->getGLTF()),
    m_scene(m_gltf > 0 ? NScene::Load(m_ok->getGLTFBase(), m_ok->getGLTFName(), m_ok->getGLTFConfig()) : NULL),
    m_num_nd(m_scene ? m_scene->getNumNd() : -1),
    m_targetnode(m_scene ? m_scene->getTargetNode() : 0),
    m_geolib(new GGeoLib(m_ok)),
    m_nodelib(new GNodeLib(m_ok, false, m_targetnode, "analytic/GScene/GNodeLib")),

    m_sensor_list(ggeo->getSensorList()),
    m_tri_geolib(ggeo->getTriGeoLib()),
    m_tri_mm0(m_tri_geolib->getMergedMesh(0)),

    m_tri_nodelib(ggeo->getNodeLib()),
    m_tri_bndlib(ggeo->getBndLib()),
    m_tri_meshindex(ggeo->getMeshIndex()),

    m_verbosity(m_scene ? m_scene->getVerbosity() : 0),
    m_root(NULL),
    m_selected_count(0)
{
    init();
}

GGeoLib* GScene::getGeoLib()
{
    return m_geolib ; 
}
guint4 GScene::getNodeInfo(unsigned idx) const 
{
     return m_tri_mm0->getNodeInfo(m_targetnode + idx);
}
guint4 GScene::getIdentity(unsigned idx) const 
{
     return m_tri_mm0->getIdentity(m_targetnode + idx);
}


void GScene::init()
{
    if(!m_scene)
    {
        LOG(fatal) << "NScene::Load FAILED" ;
        return ; 
    }

    m_ok->setVerbosity(m_verbosity);

    if(m_verbosity > 0)
    LOG(info) << "GScene::init START" ;


    if(m_gltf == 4)  assert(0 && "GScene::init early exit for gltf==4" );

    //modifyGeometry();  // try skipping the clear

    importMeshes(m_scene);
    if(m_verbosity > 1)
    dumpMeshes();


    m_root = createVolumeTree(m_scene) ;
    assert(m_root);

    if(m_verbosity > 0)
    LOG(info) << "GScene::init createVolumeTrue selected_count " << m_selected_count ; 


    // check consistency of the level transforms
    deltacheck_r(m_root, 0);

    m_nodelib->save();

    compareTrees();

    if(m_gltf == 44)  assert(0 && "GScene::init early exit for gltf==44" );


    makeMergedMeshAndInstancedBuffers() ; // <-- merging meshes requires boundaries to be set 

    checkMergedMeshes();

    if(m_gltf == 444)  assert(0 && "GScene::init early exit for gltf==444" );

    if(m_verbosity > 0)
    LOG(info) << "GScene::init DONE"

                ;

}



void GScene::dumpTriInfo() const 
{
    LOG(info) << "GScene::dumpTriInfo" 
              << " num_nd " << m_num_nd
              << " targetnode " << m_targetnode
              ;
    
    std::cout << " tri (geolib)  " << m_tri_geolib->desc() << std::endl ; 
    std::cout << " tri (nodelib) " << m_tri_nodelib->desc() << std::endl ; 

    unsigned nidx = m_num_nd ; // <-- from NScene
    for(unsigned idx = 0 ; idx < nidx ; ++idx)
    {
        guint4 id = getIdentity(idx);
        guint4 ni = getNodeInfo(idx);
        std::cout
                  << " " << std::setw(5) << idx 
                  << " " << std::setw(5) << idx + m_targetnode
                  << " ID(nd/ms/bd/sn) " << id.description() 
                  << " NI(nf/nv/ix/px) " << ni.description() 
                  << std::endl
                   ; 
    }
}


void GScene::compareTrees() const 
{
    if(m_verbosity > 1)
    {
        LOG(info) << "nodelib (GSolid) volumes " ; 
        std::cout << " ana " << m_nodelib->desc() << std::endl ; 
        std::cout << " tri " << m_tri_nodelib->desc() << std::endl ; 
    }
}


void GScene::modifyGeometry()
{
    // clear out the G4DAE geometry GMergedMesh, typically loaded from cache 
    m_geolib->clear();
}



unsigned GScene::findTriMeshIndex(const char* soname) const 
{
   unsigned missing = std::numeric_limits<unsigned>::max() ;
   unsigned tri_mesh_idx = m_tri_meshindex->getIndexSource( soname, missing );
   assert( tri_mesh_idx != missing );
   return tri_mesh_idx ; 
}



void GScene::importMeshes(NScene* scene)  // load analytic polygonized GMesh instances into m_meshes vector
{
    unsigned num_meshes = scene->getNumMeshes();
    LOG(info) << "GScene::importMeshes START num_meshes " << num_meshes  ; 
     
    for(unsigned mesh_idx=0 ; mesh_idx < num_meshes ; mesh_idx++)
    {
        NCSG* csg = scene->getCSG(mesh_idx);
        NTrianglesNPY* tris = csg->getTris();
        assert(tris);
        assert( csg->getIndex() == mesh_idx) ;

        std::string soname = csg->soname();
        unsigned tri_mesh_idx = findTriMeshIndex(soname.c_str());

        m_rel2abs_mesh[mesh_idx] = tri_mesh_idx ;  
        m_abs2rel_mesh[tri_mesh_idx] = mesh_idx ;  

        LOG(info) 
             << " mesh_idx " <<  std::setw(4) << mesh_idx
             << " tri_mesh_idx " <<  std::setw(4) << tri_mesh_idx
             << " soname " << soname 
             ;

        // hmm: absolute or relative mesh indexing ???

        GMesh* mesh = GMesh::make_mesh(tris->getTris(), mesh_idx );

        m_meshes[mesh_idx] = mesh ;
 
        assert(mesh);
    }
    LOG(info) << "GScene::importMeshes DONE num_meshes " << num_meshes  ; 
}


unsigned GScene::getNumMeshes() 
{
   return m_meshes.size();
}
GMesh* GScene::getMesh(unsigned r)
{
    assert( r < m_meshes.size() );
    return m_meshes[r];
}
NCSG* GScene::getCSG(unsigned r) 
{
    return m_scene->getCSG(r);
}



void GScene::dumpMeshes()
{
    unsigned num_meshes = getNumMeshes() ; 
    LOG(info) << "GScene::dumpMeshes" 
              << " verbosity " << m_verbosity 
              << " num_meshes " << num_meshes 
              ;

    for(unsigned r=0 ; r < num_meshes ; r++)
    {
         unsigned a = m_rel2abs_mesh[r] ; 
         unsigned a2r = m_abs2rel_mesh[a] ; 

         GMesh* mesh = getMesh( r );
         gbbox bb = mesh->getBBox(0) ; 

         std::cout 
                   << " r " << std::setw(4) << r
                   << " a " << std::setw(4) << a
                   << " "
                   << bb.description()
                   << std::endl ; 


         assert( a2r == r );
    }
}








GSolid* GScene::createVolumeTree(NScene* scene) // creates analytic GSolid/GNode tree without access to triangulated GGeo info
{
    if(m_verbosity > 0)
    LOG(info) << "GScene::createVolumeTree START verbosity " << m_verbosity  ; 
    assert(scene);

    //scene->dumpNdTree("GScene::createVolumeTree");

    nd* root_nd = scene->getRoot() ;
    assert(root_nd->idx == 0 );

    GSolid* root = createVolumeTree_r( root_nd, NULL );
    assert(root);

    assert( m_nodes.size() == scene->getNumNd()) ;

    if(m_verbosity > 0)
    LOG(info) << "GScene::createVolumeTree DONE num_nodes: " << m_nodes.size()  ; 
    return root ; 
}


GSolid* GScene::createVolumeTree_r(nd* n, GSolid* parent)
{
    guint4 id = getIdentity(n->idx);
    guint4 ni = getNodeInfo(n->idx);

    unsigned aidx = n->idx + m_targetnode ;           // absolute nd index, fed directly into GSolid index
    unsigned pidx = parent ? parent->getIndex() : 0 ; // partial parent index

    if(m_verbosity > 4)
    std::cout
           << "GScene::createVolumeTree_r"
           << " idx " << std::setw(5) << n->idx 
           << " aidx " << std::setw(5) << aidx
           << " pidx " << std::setw(5) << pidx
           << " ID(nd/ms/bd/sn) " << id.description() 
           << " NI(nf/nv/ix/px) " << ni.description() 
           << std::endl
           ; 

    // constrain node indices 
    assert( aidx == id.x && aidx == ni.z );
    if( pidx > 0)
    {
        //assert( pidx == ni.w );   // absolute node indexing 
        assert( pidx + m_targetnode == ni.w );  // relative node indexing
    }



    GSolid* node = createVolume(n);
    node->setParent(parent) ;   // tree hookup 

    typedef std::vector<nd*> VN ; 
    for(VN::const_iterator it=n->children.begin() ; it != n->children.end() ; it++)
    {
        nd* cn = *it ; 
        GSolid* child = createVolumeTree_r(cn, node);
        node->addChild(child);
    } 
    return node  ; 
}


GSolid* GScene::getNode(unsigned node_idx)
{
   // TODO: migrate to using nodelib 
    assert(node_idx < m_nodes.size());
    return m_nodes[node_idx];  
}


GSolid* GScene::createVolume(nd* n) // compare with AssimpGGeo::convertStructureVisit
{
    assert(n);
    unsigned rel_node_idx = n->idx ;
    unsigned abs_node_idx = n->idx + m_targetnode  ;  

    unsigned rel_mesh_idx = n->mesh ;   
    unsigned abs_mesh_idx = m_rel2abs_mesh[rel_mesh_idx] ;   

    GMesh* mesh = getMesh(rel_mesh_idx);
    NCSG*   csg =  getCSG(rel_mesh_idx);

    glm::mat4 xf_global = n->gtransform->t ;    
    glm::mat4 xf_local  = n->transform->t ;    
    GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
    GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));

    GSolid* solid = new GSolid( rel_node_idx, gtransform, mesh, UINT_MAX, NULL );     
   
    solid->setLevelTransform(ltransform); 

    transferMetadata( solid, csg, n ); 
    transferIdentity( solid, n ); 

    std::string bndspec = lookupBoundarySpec(solid, n);  // using just transferred boundary from tri branch

    GParts* pts = GParts::make( csg, bndspec.c_str(), m_verbosity  ); // amplification from mesh level to node level 

    pts->setBndLib(m_tri_bndlib);

    solid->setParts( pts );


    if(m_verbosity > 2) 
    LOG(info) << "GScene::createVolume"
              << " verbosity " << m_verbosity
              << " rel_node_idx " << std::setw(5) << rel_node_idx 
              << " abs_node_idx " << std::setw(6) << abs_node_idx 
              << " rel_mesh_idx " << std::setw(3) << rel_mesh_idx 
              << " abs_mesh_idx " << std::setw(3) << abs_mesh_idx 
              << " ridx " << std::setw(3) << n->repeatIdx
              << " solid " << solid
              << " solid.pts " << pts 
              << " solid.idx " << solid->getIndex()
              << " solid.lvn " << solid->getLVName()
              << " solid.pvn " << solid->getPVName()
              ;


    addNode(solid, n );

    return solid ; 
}


void GScene::transferMetadata( GSolid* node, const NCSG* csg, const nd* n)
{
    assert(n->repeatIdx > -1);

    node->setRepeatIndex( n->repeatIdx ); 
    node->setCSGFlag( csg->getRootType() );
    node->setCSGSkip( csg->isSkip() );

    std::string pvname = n->pvname  ;  // pv from the node, not the csg/mesh
    std::string lvn = csg->lvname()  ;

    node->setPVName( pvname.c_str() );
    node->setLVName( lvn.c_str() );

    bool selected = n->selected > 0 ;
    node->setSelected( selected  );

    if(selected) m_selected_count++ ; 
}


void GScene::transferIdentity( GSolid* node, const nd* n)
{
    // passing tri identity into analytic branch 
    unsigned rel_node_idx = n->idx ;
    unsigned abs_node_idx = n->idx + m_targetnode  ;  

    unsigned rel_mesh_idx = n->mesh ;   
    unsigned abs_mesh_idx = m_rel2abs_mesh[rel_mesh_idx] ;   



    guint4 tri_id = getIdentity(n->idx);  // offsets internally 

    unsigned tri_nodeIdx          = tri_id.x ;  // full geometry absolute
    unsigned tri_meshIdx          = tri_id.y ;  // absolute (assimp) G4DAE mesh index
    unsigned tri_boundaryIdx      = tri_id.z ; 
    unsigned tri_sensorSurfaceIdx = tri_id.w ; 

    NSensor* tri_sensor = m_sensor_list->findSensorForNode(tri_nodeIdx);
/*
    //  All 5 nodes of the PMT have associated NSensor but only cathode has non-zero index
    if(tri_sensor) std::cout << "got sensor " 
                             << " tri_nodeIdx " << tri_nodeIdx
                             << " tri_sensorSurfaceIdx " << tri_sensorSurfaceIdx  
                             << std::endl ; 
*/

    node->setBoundary(  tri_boundaryIdx ); 
    node->setSensor(    tri_sensor );      
    node->setSensorSurfaceIndex( tri_sensorSurfaceIdx );


    guint4 check_id = node->getIdentity();

    //bool match_node_index     = check_id.x == tri_id.x ;
    //bool match_mesh_index     = check_id.y == tri_id.y ;
    bool match_boundary_index = check_id.z == tri_id.z ;
    bool match_sensor_index   = check_id.w == tri_id.w ;

    //assert( match_node_index );    
    //assert( match_mesh_index );
    assert( match_boundary_index );
    assert( match_sensor_index );


    assert( rel_node_idx == node->getIndex() );
    assert( abs_node_idx == tri_nodeIdx );

    assert( tri_meshIdx == abs_mesh_idx );
    assert( check_id.y  == rel_mesh_idx );

/*
    if(!match_mesh_index)   // how is mesh idx used ?? does is need to be absolute ??
        LOG(info) 
           << " match_mesh_index "
           << " check_id.y " << check_id.y
           << " tri_id.y " << tri_id.y
           << " tri_meshIdx " << tri_meshIdx
           << " rel_mesh_idx " << rel_mesh_idx
           << " abs_mesh_idx " << abs_mesh_idx
           ; 
*/

    if(!match_sensor_index)
        LOG(info) << " match_sensor_index "
                  << " check_id.w  " << check_id.w 
                  << " tri_id.w " << tri_id.w
                  ;
    
}




std::string GScene::lookupBoundarySpec( const GSolid* node, const nd* n) const 
{
    unsigned tri_boundary = node->getBoundary();    // get the just transferred tri_boundary 

    guint4 ana_bnd = m_tri_bndlib->parse( n->boundary.c_str());  // NO SURFACES
    guint4 tri_bnd = m_tri_bndlib->getBnd(tri_boundary);

    assert( ana_bnd.x == tri_bnd.x && "imat should match");  
    assert( ana_bnd.w == tri_bnd.w && "omat should match");


    std::string tri_bndname = m_tri_bndlib->shortname(tri_bnd);

    const char* tri_spec = tri_bndname.c_str();
    
    if(m_verbosity > 3)
    std::cout  
              << " tri_boundary " << tri_boundary
              << " tri_bnd " << tri_bnd.description()
              << " ana_bnd " << ana_bnd.description()
              << " tri_spec " << tri_spec
              << " n.boundary " << n->boundary 
              << std::endl 
              ;
  
    return tri_bndname ; 
}







void GScene::addNode(GSolid* node, nd* n)
{
    unsigned node_idx = n->idx ;
    assert(m_nodes.count(node_idx) == 0); 
    m_nodes[node_idx] = node ; 

    // TODO ... get rid of above, use the nodelib 
    m_nodelib->add(node);    
}




void GScene::deltacheck_r( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GMatrixF* gtransform = solid->getTransform();

    //GMatrixF* ltransform = solid->getLevelTransform();  
    GMatrixF* ctransform = solid->calculateTransform();
    float delta = gtransform->largestDiff(*ctransform);

    if(m_verbosity > 1)
    std::cout << "GScene::deltacheck_r gtransform " << gtransform->brief(7) << std::endl  ;

    assert(delta < 1e-6) ;

    for(unsigned int i = 0; i < node->getNumChildren(); i++) deltacheck_r(node->getChild(i), depth + 1 );
}





void GScene::makeMergedMeshAndInstancedBuffers()   // using m_geolib to makeMergedMesh
{
    unsigned num_repeats = m_scene->getNumRepeats(); // global 0 included
    unsigned nmm_created = 0 ; 

    if(m_verbosity > 0)
    LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers num_repeats " << num_repeats << " START " ;  


    for(unsigned ridx=0 ; ridx < num_repeats ; ridx++)
    {
         if(m_verbosity > 1)
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

         GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, base, m_root, m_verbosity );   

         assert(mm);

         makeInstancedBuffers(mm, ridx);

         mm->setGeoCode(OpticksConst::GEOCODE_ANALYTIC);

         if(m_verbosity > 1)
         std::cout << "GScene::makeMergedMeshAndInstancedBuffers"
                   << " ridx " << ridx 
                   << " mm " << mm->getIndex()
                   << " nmm_created " << nmm_created
                   << std::endl ; 

         nmm_created++ ; 

         GMergedMesh* mmc = m_geolib->getMergedMesh(ridx);
         assert(mmc == mm);

         if(m_verbosity > 1)
         LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers ridx " << ridx << " DONE " ;  
    }

    unsigned nmm = m_geolib->getNumMergedMesh();
   
     
    if(m_verbosity > 0)
    LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers DONE"
              << " num_repeats " << num_repeats
              << " nmm_created " << nmm_created
              << " nmm " << nmm
               ; 

    assert(nmm == nmm_created);


}


void GScene::checkMergedMeshes()
{
    int nmm = m_geolib->getNumMergedMesh();
    int mia = 0 ;

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_geolib->getMergedMesh(i);
        if(m_verbosity > 2) 
        std::cout << "GScene::checkMergedMeshes i:" << std::setw(4) << i << " mm? " << (mm ? int(mm->getIndex()) : -1 ) << std::endl ; 
        if(mm == NULL) mia++ ; 
    } 

    if(m_verbosity > 2 || mia != 0)
    LOG(info) << "GScene::checkMergedMeshes" 
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

    if(m_verbosity > 1) 
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


