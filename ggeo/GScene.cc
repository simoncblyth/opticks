#include "OpticksConst.hh"



#include "SSortKV.hh"

#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NScene.hpp"
#include "NNode.hpp"
#include "NSensor.hpp"
#include "NCSG.hpp"
#include "Nd.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NSensorList.hpp"

#include "OpticksQuery.hh"
#include "OpticksEvent.hh"
#include "Opticks.hh"

#include "GItemList.hh"
#include "GItemIndex.hh"
#include "GGeo.hh"
#include "GParts.hh"
#include "GGeoLib.hh"
#include "GNodeLib.hh"
#include "GMeshLib.hh"
#include "GBndLib.hh"
#include "GMatrix.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GScene.hh"
#include "GColorizer.hh"
#include "GMergedMesh.hh"

#define AS_VEC3(q) glm::vec3((q).x, (q).y, (q).z) 

#include "PLOG.hh"

GGeoLib*          GScene::getGeoLib() {          return m_geolib ; } 

GBndLib*          GScene::getBndLib() {          return m_ggeo->getBndLib(); } 



const char*       GScene::getIdentifier(){       return "GScene" ;  }
GScintillatorLib* GScene::getScintillatorLib() { return m_ggeo->getScintillatorLib(); } 
GSourceLib*       GScene::getSourceLib() {       return m_ggeo->getSourceLib(); } 



void GScene::save() const 
{
    m_geolib->dump("GScene::save");

    m_geolib->save();
    m_nodelib->save();
    m_meshlib->save();
}


GScene* GScene::Load(Opticks* ok, GGeo* ggeo)
{
    bool loaded = true ; 
    GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
    return scene ;  
}


GScene::GScene( Opticks* ok, GGeo* ggeo, bool loaded )
    :
    GGeoBase(),
    m_ok(ok),
    m_query(ok->getQuery()),
    m_ggeo(ggeo),

    m_sensor_list(ggeo->getSensorList()),
    m_tri_geolib(ggeo->getGeoLib()),
    m_tri_mm0(m_tri_geolib->getMergedMesh(0)),

    m_tri_nodelib(ggeo->getNodeLib()),
    m_tri_bndlib(ggeo->getBndLib()),
    m_tri_meshlib(ggeo->getMeshLib()),
    m_tri_meshindex(m_tri_meshlib->getMeshIndex()),


    m_analytic(true),
    m_loaded(loaded),
    m_honour_selection(true),
    m_gltf(m_ok->getGLTF()),
    m_scene_config( m_ok->getSceneConfig() ),
    m_scene(loaded ? NULL : (m_gltf > 0 ? NScene::Load(m_ok->getGLTFBase(), m_ok->getGLTFName(), m_scene_config, m_ok->getDbgNode()) : NULL)),
    m_num_nd(m_scene ? m_scene->getNumNd() : -1),
    m_targetnode(m_scene ? m_scene->getTargetNode() : 0),

    m_geolib(loaded ? GGeoLib::Load(m_ok, m_analytic, m_tri_bndlib )   : new GGeoLib(m_ok, m_analytic, m_tri_bndlib)),
    m_nodelib(loaded ? GNodeLib::Load(m_ok, m_analytic ) : new GNodeLib(m_ok, m_analytic)),
    m_meshlib(loaded ? GMeshLib::Load(m_ok, m_analytic)  : new GMeshLib(m_ok, m_analytic)),

    m_colorizer(new GColorizer(m_nodelib, m_geolib, m_tri_bndlib, ggeo->getColors(), GColorizer::PSYCHEDELIC_NODE )),   // GColorizer::SURFACE_INDEX

    m_verbosity(m_scene ? m_scene->getVerbosity() : 0),
    m_root(NULL),
    m_selected_count(0)
{


    if(!m_loaded)
    {
        initFromGLTF();
    }
}





void GScene::dumpNode( unsigned nidx)
{
    if(!m_scene) return ; 

    nd* n = m_scene->getNd(nidx);
    LOG(info) << "GScene::dump_node" 
              << " nidx " << std::setw(6) << nidx
              << ( n ? " FOUND " : " NO-SUCH-NODE " )
              ;



    if(n)
        std::cout << std::endl << n->detail() << std::endl ;  
}




GNodeLib* GScene::getNodeLib()
{
    return m_nodelib ; 
}

guint4 GScene::getNodeInfo(unsigned idx) const 
{
     return m_tri_mm0->getNodeInfo(m_targetnode + idx);
}
guint4 GScene::getIdentity(unsigned idx) const 
{
     return m_tri_mm0->getIdentity(m_targetnode + idx);
}


GMergedMesh* GScene::getMergedMesh(unsigned idx)
{
    return m_geolib->getMergedMesh(idx);
}
GSolid* GScene::getSolid(unsigned idx)
{
    return m_nodelib->getSolid(idx);
}


void GScene::initFromGLTF()
{
    if(!m_scene)
    {
        LOG(fatal) << "NScene::Load FAILED" ;
        return ; 
    }

    m_ok->setVerbosity(m_verbosity);

    //if(m_verbosity > 0)
    LOG(info) << "GScene::init START" ;



    //modifyGeometry();  // try skipping the clear

    importMeshes(m_scene);

    if(m_verbosity > 1)
    dumpMeshes();

    compareMeshes();

    if(m_gltf == 4 || m_gltf == 44)  assert(0 && "GScene::init early exit for gltf==4 or gltf==44" );

    m_root = createVolumeTree(m_scene) ;
    assert(m_root);

    if(m_verbosity > 0)
    LOG(info) << "GScene::init createVolumeTrue selected_count " << m_selected_count ; 


    // check consistency of the level transforms
    deltacheck_r(m_root, 0);

    //m_nodelib->save();

    compareTrees();

    if(m_gltf == 44)  assert(0 && "GScene::init early exit for gltf==44" );


    makeMergedMeshAndInstancedBuffers() ; // <-- merging meshes requires boundaries to be set 

    checkMergedMeshes();

    prepareVertexColors();


    save(); 

    if(m_gltf == 444)  assert(0 && "GScene::init early exit for gltf==444" );

    //if(m_verbosity > 0)
    LOG(info) << "GScene::init DONE" ;

}


void GScene::prepareVertexColors()
{
    LOG(info) << "GScene::prepareVertexColors START" ;
    m_colorizer->writeVertexColors();
    LOG(info) << "GScene::prepareVertexColors DONE " ;
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


    //m_tri_mm0->dumpSolids();
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

        // rel2abs/abs2rel are not good names
        //
        // analytic/sc.py GDML branch is using mesh indexing based on lvIdx unlike G4DAE, 
        // so need solid name mapping 
        // to establish correspondence, the names include pointer addresses so this ensures
        // that the source GDML and G4DAE files were produced by a single process

        m_rel2abs_mesh[mesh_idx] = tri_mesh_idx ;  
        m_abs2rel_mesh[tri_mesh_idx] = mesh_idx ;  

        if(m_verbosity > 2)
        LOG(info) 
             << " mesh_idx " <<  std::setw(4) << mesh_idx
             << " tri_mesh_idx " <<  std::setw(4) << tri_mesh_idx
             << " soname " << soname 
             ;

        GMesh* mesh = GMesh::make_mesh(tris->getTris(), mesh_idx );
        assert(mesh);

        mesh->setCSG(csg);
        mesh->setName(strdup(soname.c_str()));

        const char* aname = mesh->getName() ; 
        GMesh* alt = m_tri_meshlib->getMesh(aname, false);
        assert(alt);
        mesh->setAlt(alt);

        alt->setAlt(mesh);



        //m_meshes[mesh_idx] = mesh ;
        m_meshlib->add(mesh);
    }
    LOG(info) << "GScene::importMeshes DONE num_meshes " << num_meshes  ; 
}




unsigned GScene::getNumMeshes() 
{
   return m_meshlib->getNumMeshes() ;
}
GMesh* GScene::getMesh(unsigned r)
{
    return m_meshlib->getMesh(r) ; 
}
NCSG* GScene::getCSG(unsigned r) 
{
    return m_scene->getCSG(r);
}

NCSG* GScene::findCSG(const char* soname, bool startswith) const 
{
    return m_scene->findCSG(soname, startswith);
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

void GScene::compareMeshes()
{
    compareMeshes_GMeshBB();
}


nbbox GScene::getBBox(const char* soname, NSceneConfigBBoxType bbty) const 
{
    nbbox ubb ; 
    const GMeshLib* ana = m_meshlib ; 
    const GMeshLib* tri = m_tri_meshlib ; 

    bool startswith = false ; 
    NCSG* csg = findCSG(soname, startswith); 

    if( bbty == CSG_BBOX_ANALYTIC )
    {
        nbbox abb = csg->bbox_analytic();         // depends on my CSG tree primitive/composite bbox calc 
        ubb.copy_from(abb);
    }
    else if(bbty == CSG_BBOX_PARSURF )
    {
        nbbox sbb = csg->bbox_surface_points() ;  // depends on parametric surface points and their composite SDF epsilon selection
        ubb.copy_from(sbb);
    }
    else if(bbty == CSG_BBOX_POLY )
    {
        GMesh* a = ana->getMesh(soname, startswith);
        gbbox _pbb = a->getBBox(0) ;   // depends on my NPolygonization IM/HY/DCS/etc... 
        nbbox pbb = make_bbox( _pbb.min.x, _pbb.min.y, _pbb.min.z, _pbb.max.x, _pbb.max.y, _pbb.max.z );
        ubb.copy_from(pbb);
    }
    else if(bbty == CSG_BBOX_G4POLY )
    {
        GMesh* g4poly = tri->getMesh(soname, startswith);
        gbbox _bbb = g4poly->getBBox(0) ;   // depends on G4 polygonization 
        nbbox g4bb = make_bbox( _bbb.min.x, _bbb.min.y, _bbb.min.z, _bbb.max.x, _bbb.max.y, _bbb.max.z );
        ubb.copy_from(g4bb);
    }
    return ubb ; 
}


void GScene::compareMeshes_GMeshBB()
{
    GMeshLib* ana = m_meshlib ; 
    GMeshLib* tri = m_tri_meshlib ; 

    unsigned num_meshes = ana->getNumMeshes() ;
    unsigned num_meshes2 = tri->getNumMeshes() ;
    assert( num_meshes == num_meshes2 );

    float cut = 0.1 ;  // mm

    unsigned num_discrepant(0);

    bool startswith = false ; 
    bool descending = true ; 
    bool present_bb = m_gltf > 4  ; 

    SSortKV delta(descending);

    NSceneConfigBBoxType bbty = m_scene->bbox_type() ; 
    const char* bbty_ = m_scene->bbox_type_string() ;

    LOG(info) << "GScene::compareMeshes_GMeshBB"
              << " num_meshes " << num_meshes
              << " cut " << cut 
              << " bbty " << bbty_
              << " parsurf_level " << m_scene_config->parsurf_level 
              << " parsurf_target " << m_scene_config->parsurf_target 
              ;

    for(unsigned i=0 ; i < num_meshes ; i++ )
    {
        GMesh* a = ana->getMesh(i);
        unsigned mesh_idx = a->getIndex();
        assert( mesh_idx == i );
        const char* soname = a->getName() ; 

        nbbox okbb = getBBox(soname, bbty );
        nbbox g4bb = getBBox(soname, CSG_BBOX_G4POLY );

        float diff = nbbox::MaxDiff( okbb, g4bb) ;

        delta.add(soname, diff);
    }

    delta.sort();

    for(unsigned i=0 ; i < delta.getNum() ; i++)
    {
        float dmax  = delta.getVal(i);
        if(dmax < cut) continue ; 
        
        num_discrepant++ ;

        std::string name = delta.getKey(i);
        const char* soname = name.c_str(); 

        GMesh* a = ana->getMesh(soname, startswith);
        GMesh* b = tri->getMesh(soname, startswith);

        const GMesh* a2 = b->getAlt(); assert( a2 == a );
        const GMesh* b2 = a->getAlt(); assert( b2 == b );

        nbbox okbb = getBBox(soname, bbty );
        nbbox g4bb = getBBox(soname, CSG_BBOX_G4POLY );

        NCSG*  csg = findCSG(soname, startswith); assert( csg == a->getCSG() );  
        unsigned a_mesh_id = a->getIndex();
        int lvidx = m_scene->lvidx(a_mesh_id);
        nnode* root = csg->getRoot();

        std::vector<unsigned> nodes ; 
        m_scene->collect_mesh_nodes(nodes, a_mesh_id );
        std::string nodes_ = m_scene->present_mesh_nodes(nodes, 10) ; 

        assert( nodes.size() > 0 );

        unsigned nsp = csg->getNumSurfacePoints();

        std::string typemask_ = root->get_type_mask_string() ; 

        glm::vec3 dmn = okbb.min - g4bb.min ; 
        glm::vec3 dmx = okbb.max - g4bb.max ; 


        std::cout 
               << std::setw(10) << dmax
               << std::setw(40) << name
               << " lvidx " << std::setw(3) << lvidx 
               << " nsp " << std::setw(6) << nsp
               ;

        if(present_bb)
        {
            std::cout 
               << " amn " << gpresent(okbb.min)
               << " bmn " << gpresent(g4bb.min)
               << " dmn " << gpresent(dmn)
               << " amx " << gpresent(okbb.max)
               << " bmx " << gpresent(g4bb.max)
               << " dmx " << gpresent(dmx)
               ;
        }
        else
        {
            std::cout 
               << " " << std::setw(50) << typemask_ 
               << " " << nodes_
               ;
        } 
        std::cout << std::endl ;
    }

    LOG(info) << "GScene::compareMeshes_GMeshBB"
              << " num_meshes " << num_meshes
              << " cut " << cut 
              << " bbty " << bbty_
              << " num_discrepant " << num_discrepant
              << " frac " << float(num_discrepant)/float(num_meshes)
              ;
}





GSolid* GScene::createVolumeTree(NScene* scene) // creates analytic GSolid/GNode tree without access to triangulated GGeo info
{
    if(m_verbosity > 0)
    LOG(info) << "GScene::createVolumeTree START"
              << "  verbosity " << m_verbosity  
              << " query " << m_query->description()
              ; 
    assert(scene);

    //scene->dumpNdTree("GScene::createVolumeTree");

    nd* root_nd = scene->getRoot() ;
    assert(root_nd->idx == 0 );

    GSolid* parent = NULL ;
    unsigned depth = 0 ; 
    bool recursive_select = false ; 
    GSolid* root = createVolumeTree_r( root_nd, parent, depth, recursive_select );
    assert(root);

    assert( m_nodes.size() == scene->getNumNd()) ;

    if(m_verbosity > 0)
    LOG(info) << "GScene::createVolumeTree DONE num_nodes: " << m_nodes.size()  ; 
    return root ; 
}


GSolid* GScene::createVolumeTree_r(nd* n, GSolid* parent, unsigned depth, bool recursive_select  )
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

    GSolid* node = createVolume(n, depth, recursive_select );
    node->setParent(parent) ;   // tree hookup 


    typedef std::vector<nd*> VN ; 
    for(VN::const_iterator it=n->children.begin() ; it != n->children.end() ; it++)
    {
        nd* cn = *it ; 
        GSolid* child = createVolumeTree_r(cn, node, depth+1, recursive_select );
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


GSolid* GScene::createVolume(nd* n, unsigned depth, bool& recursive_select  ) // compare with AssimpGGeo::convertStructureVisit
{
    assert(n);
    unsigned rel_node_idx = n->idx ;
    unsigned abs_node_idx = n->idx + m_targetnode  ;  
    assert(m_targetnode == 0);

    unsigned rel_mesh_idx = n->mesh ;   
    unsigned abs_mesh_idx = m_rel2abs_mesh[rel_mesh_idx] ;   

    GMesh* mesh = getMesh(rel_mesh_idx);
    const GMesh* altmesh = mesh->getAlt();
    assert(altmesh);


    NCSG*   csg =  getCSG(rel_mesh_idx);

    glm::mat4 xf_global = n->gtransform->t ;    
    glm::mat4 xf_local  = n->transform->t ;    
    GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
    GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));

    // for odd gltf : use the tri GMesh within the analytic GSolid 
    // for direct comparison of analytic ray trace with tri polygonization

    GSolid* solid = new GSolid( rel_node_idx, gtransform, (m_gltf == 3 ? altmesh : mesh ), UINT_MAX, NULL );     
   
    solid->setLevelTransform(ltransform); 

    transferMetadata( solid, csg, n, depth, recursive_select ); 
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

void GScene::transferMetadata( GSolid* node, const NCSG* csg, const nd* n, unsigned depth, bool& recursive_select )
{
    assert(n->repeatIdx > -1);

    node->setRepeatIndex( n->repeatIdx ); 
    node->setCSGFlag( csg->getRootType() );
    node->setCSGSkip( csg->isSkip() );

    std::string pvname = n->pvname  ;  // pv from the node, not the csg/mesh
    std::string lvn = csg->lvname()  ;

    node->setPVName( pvname.c_str() );
    node->setLVName( lvn.c_str() );

    //bool selected = n->selected > 0 ;
    //node->setSelected( selected  );
    // more convenient to do selection here than in the python gltf preparation

    bool selected = m_query->selected(pvname.c_str(), n->idx, depth, recursive_select); 

    node->setSelected( selected  );

    if(selected) m_selected_count++ ; 

/*
    LOG(info) << "GScene::transferMetadata"
              << " idx " << std::setw(6) << n->idx
              << " depth " << std::setw(6) << depth
              << " recursive_select " << ( recursive_select ? "Y" : "N" )
              << " selected " << ( selected ? "Y" : "N" )
              ;
*/



}


void GScene::transferIdentity( GSolid* node, const nd* n)
{
    // passing tri identity into analytic branch 
    unsigned rel_node_idx = n->idx ;
    unsigned abs_node_idx = n->idx + m_targetnode  ;  
    assert(m_targetnode == 0);

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

    if(m_gltf == 3)
    {
       // using tri mesh within ana
        assert( check_id.y  == abs_mesh_idx );
    }
    else
    {
        assert( check_id.y  == rel_mesh_idx );
    }


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

    guint4 tri = m_tri_bndlib->getBnd(tri_boundary);
    guint4 ana = m_tri_bndlib->parse( n->boundary.c_str());  // NO SURFACES

 
    //assert( ana.x == tri.x && "imat should match");  
    //assert( ana.w == tri.w && "omat should match");

    std::string ana_spec = m_tri_bndlib->shortname(ana);
    std::string tri_spec = m_tri_bndlib->shortname(tri);
    std::string spec = tri_spec ; 



    if( !(ana.x == tri.x && ana.w == tri.w) )
    {
         LOG(warning) << "GScene::lookupBoundarySpec ana/tri imat/omat MISMATCH "
                      << " tri " << tri.description()
                      << " ana " << ana.description()
                      << " tri_spec " << tri_spec
                      << " ana_spec " << ana_spec
                      << " spec " << spec
                      ;
    } 
   
/*
    if(n->selected)
    {
        //LOG(warning) << " using ana_spec from n.boundary " << n->boundary <<  " for selected node " << n->idx  ; 
        spec = ana_spec ; 
    }
*/ 


/*
    if(m_verbosity > 3  || n->selected)
    std::cout  
              << " nidx " << std::setw(5) << n->idx
              << " tri_boundary " << tri_boundary
              << " tri " << tri.description()
              << " ana " << ana.description()
              << " tri_spec " << tri_spec
              << " ana_spec " << ana_spec
              << " n.boundary " << n->boundary 
              << " USING spec : " << spec 
              << std::endl 
              ;
*/  


    return spec ; 
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
    unsigned num_repeats = m_scene->getNumRepeats() ; //  number of non-zero ridx instances, typically small eg 4 for j1707

    unsigned num_ridx = 1u + num_repeats ; 

    unsigned nmm_created = 0 ; 

    //if(m_verbosity > 0)
    LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers.START "
              << "  num_repeats " << num_repeats 
              << "  num_ridx " << num_ridx 
              ;  

    bool honour_selection = false ;   // in order to match GTreeCheck 


    for(unsigned ridx=0 ; ridx < num_ridx ; ridx++)
    {
         if(m_verbosity > 1)
         LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers ridx " << ridx << " START " ;  

         bool inside = ridx == 0 ? false : false ; 

         const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside, honour_selection );

         assert( instances.size() > 0u );

         GSolid* instance0 = dynamic_cast<GSolid*>(instances[0]) ; 

         if(ridx == 0 )
         {
             assert( instances.size() == 1 && instance0 == m_root );
         }     
   

         GSolid* base = ridx == 0 ? NULL : instance0 ; 

         GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, base, m_root, m_verbosity );   

         assert(mm);

         mm->addInstancedBuffers(instances);

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
              << " num_ridx (including global 0) " << num_ridx 
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

    if(!m_honour_selection)
    {
        assert(mia == 0 );
    }
}




/*

void GScene::makeInstancedBuffers(GMergedMesh* mm, unsigned ridx, bool honour_selection)
{
    bool inside = ridx == 0 ; 
    const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside, honour_selection );
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

NPY<unsigned>* GScene::makeInstanceIdentityBuffer(const std::vector<GNode*>& instances, unsigned ridx)   
{
    unsigned num_instances = instances.size(); 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 4);
    buf->zero(); 

    // huh : this is not doing the solid duplication of GTreeCheck::makeInstanceIdentityBuffer  ???


    for(unsigned i=0 ; i < num_instances ; i++)
    {
        GSolid* instance = dynamic_cast<GSolid*>(instances[i]) ; 
        guint4 id = instance->getIdentity();
        glm::uvec4 uid(id.x, id.y, id.z, id.w);
        buf->setQuadU( uid, i );
    }
    return buf ;  
}

NPY<unsigned>* GScene::makeAnalyticInstanceIdentityBuffer( const std::vector<GNode*>& instances, unsigned ridx) 
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

*/




void GScene::debugNodeIntersects(int dbgnode, OpticksEvent* evt)
{
    // gets invoked from OpticksHub::anaEvent 
    // for the analytic glTF branch when the *dbgnode* option is used 
    // eg "--dbgnode 3159" is used 

    GSolid* solid = m_nodelib->getSolid(dbgnode);
    GNodeLib* nlib = m_ggeo->getNodeLib();

    const GMesh* mesh = solid->getMesh();
    unsigned mesh_idx = mesh->getIndex();
    NCSG* csg = m_scene->getCSG(mesh_idx);
    nnode* root = csg->getRoot();

    std::function<float(float,float,float)> sdf = root->sdf();

    float epsilon = 0.1 ; 
    //float epsilon = 0.05 ;   //  2798/100,000

    LOG(info) << "OpticksHub::debugNode " << dbgnode << " epsilon " << epsilon  ;
    LOG(info) << " nodelib " << nlib->desc() ;
    LOG(info) << " solid " <<  ( solid ? solid->description() : " NONE " )  ; 
    LOG(info) << " csg.meta " << csg->meta() ; 
    LOG(info) << " csg.desc " << csg->desc() ; 

    assert(solid);

    // fails to find with tboolean- because thats test mode not gltf mode
    glm::mat4  gtr = solid->getTransformMat4();
    glm::mat4 igtr = glm::inverse(gtr);   

    std::cout << gpresent("gtr", gtr) << std::endl ; 
    std::cout << gpresent("igtr", igtr) << std::endl ; 


    NPY<float>* pho = evt->getPhotonData(); 
    NPY<unsigned long long>* seq = evt->getSequenceData() ;
    //pho->dump();
    //seq->dump();
    LOG(info) << " pho " << pho->getShapeString() ; 
    LOG(info) << " seq " << seq->getShapeString() ; 

    unsigned num_pho  = pho->getShape(0); 
    unsigned num_seq  = seq->getShape(0); 
    assert(num_pho == num_seq);

    unsigned long long seqhis, seqmat ; 

    //unsigned long long TO_SA =  0x8dull ;
    //unsigned long long seqhis_select = TO_SA ; 

   // unsigned count_select(0); 
   // unsigned count_excursion(0); 


    typedef std::map<unsigned long long, unsigned> MQC ;
    MQC tot ; 
    MQC exc ; 
 

    for(unsigned i=0 ; i < num_pho ; i++)
    {
        glm::vec4 post = pho->getQuad(i,0,0);

        glm::vec4 pos(post);
        pos.w = 1.0f ; 

        glm::vec4 lpos = igtr * pos ; 
        float sd = sdf(lpos.x, lpos.y, lpos.z);
        float asd = std::abs(sd) ;

        seqhis = seq->getValue(i,0,0);
        seqmat = seq->getValue(i,0,1);
        
        tot[seqhis]++;
        if(asd > epsilon) exc[seqhis]++ ;  
   }





/*
        if(seqhis == seqhis_select )
        {
            count_select++ ; 
            //if(count_select % 1000 == 0)
            {
                count_excursion++ ; 
                std::cout 
                     << " i " << std::setw(6) <<  i
                     << " c " << std::setw(6) <<  count_select
                     << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
                     << " seqmat " << std::setw(16) << std::hex << seqmat << std::dec
                  //   << " post " << glm::to_string(post)
                     << " t " << std::setw(10) << post.w
                     << " lpos " << glm::to_string(lpos)
                     << " sd " << sd
                     << std::endl 
                     ;
                 }
         }
*/


    LOG(info) << "SDF excursions by photon seqhis categories "
              << " num_pho " << std::setw(6) << num_pho
              << " epsilon " << epsilon
              ;

    /*
    typedef std::vector<unsigned long long> VQ  ; 
    VQ vq ; 
    for(MQC::const_iterator it=tot.begin() ; it != tot.end() ; it++)
    {
        vq.push_back( it->first );
    }
    */
 
    for(MQC::const_iterator it=tot.begin() ; it != tot.end() ; it++)
    {
        unsigned long long seqhis = it->first ; 
        unsigned tot_ = it->second ; 
        unsigned exc_ = exc[seqhis];
        float frac = exc_/tot_ ; 

        std::cout 
             << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
             << " tot " << std::setw(6) << tot_ 
             << " exc " << std::setw(6) << exc_
             << " exc/tot " << std::setw(6) << frac
             << std::endl ;  
    }


}



