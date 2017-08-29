postcache_analytic_gpu_gltf_option_not_working
==================================================


* Hmm there is no GScene postcache ?  Implemented postcache GScene for this


Does --gltf 3 (ie using G4 tri but Opticks ana) work postcache 
------------------------------------------------------------------

Looks like the gltf setting in force when making the cache will 
rule postcache : with no possibility to change, as its layed 
down in the GMergedMesh.

::

     644 GSolid* GScene::createVolume(nd* n, unsigned depth, bool& recursive_select  ) // compare with AssimpGGeo::convertStructureVisit
     645 {
     646     assert(n);
     647     unsigned rel_node_idx = n->idx ;
     648     unsigned abs_node_idx = n->idx + m_targetnode  ;
     649     assert(m_targetnode == 0);
     650 
     651     unsigned rel_mesh_idx = n->mesh ;
     652     unsigned abs_mesh_idx = m_rel2abs_mesh[rel_mesh_idx] ;
     653 
     654     GMesh* mesh = getMesh(rel_mesh_idx);
     655     const GMesh* altmesh = mesh->getAlt();
     656     assert(altmesh);
     657 
     658 
     659     NCSG*   csg =  getCSG(rel_mesh_idx);
     660 
     661     glm::mat4 xf_global = n->gtransform->t ;
     662     glm::mat4 xf_local  = n->transform->t ;
     663     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
     664     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));
     665 
     666     // for odd gltf : use the tri GMesh within the analytic GSolid 
     667     // for direct comparison of analytic ray trace with tri polygonization
     668 
     669     GSolid* solid = new GSolid( rel_node_idx, gtransform, (m_gltf == 3 ? altmesh : mesh ), UINT_MAX, NULL );
     670 






Analytic GPU geometry switch : --gltf 1 OR --gltf 3
----------------------------------------------------------

::

    simon:ggeo blyth$ grep m_gltf *.*
    GGeo.cc:   m_gltf(m_ok->getGLTF()),   
    GGeo.cc:    return m_gltf > 0 ? m_geolib_analytic : m_geolib ; 
    GGeo.cc:    return m_gltf > 0 ? m_nodelib_analytic : m_nodelib ; 
    GGeo.hh:        int                           m_gltf ; 
    GScene.cc:    m_gltf(m_ok->getGLTF()),
    GScene.cc:    m_scene(m_gltf > 0 ? NScene::Load(m_ok->getGLTFBase(), m_ok->getGLTFName(), m_scene_config, m_ok->getDbgNode()) : NULL),
    GScene.cc:    if(m_gltf == 4 || m_gltf == 44)  assert(0 && "GScene::init early exit for gltf==4 or gltf==44" );
    GScene.cc:    if(m_gltf == 44)  assert(0 && "GScene::init early exit for gltf==44" );
    GScene.cc:    if(m_gltf == 444)  assert(0 && "GScene::init early exit for gltf==444" );
    GScene.cc:    bool present_bb = m_gltf > 4  ; 
    GScene.cc:    GSolid* solid = new GSolid( rel_node_idx, gtransform, (m_gltf == 3 ? altmesh : mesh ), UINT_MAX, NULL );     
    GScene.cc:    if(m_gltf == 3)
    GScene.hh:        int      m_gltf ; 
    simon:ggeo blyth$ 


::

    simon:optixrap blyth$ grep GGeoBase *.*
    OGeo.cc://#include "GGeoBase.hh"
    OGeo.hh://class GGeoBase ; 
    OGeo.hh:    //GGeoBase*            m_ggeo ; 
    OScene.cc:#include "GGeoBase.hh"
    OScene.cc:    m_ggeo = m_hub->getGGeoBase();
    OScene.hh:class GGeoBase ; 
    OScene.hh:       GGeoBase*            m_ggeo ; 
    simon:optixrap blyth$ 


::

    329 GGeoBase* OpticksHub::getGGeoBase()
    330 {
    331    // analytic switch 
    332     return m_gltf ? dynamic_cast<GGeoBase*>(m_gscene) : dynamic_cast<GGeoBase*>(m_ggeo) ;
    333 }
    334 


::

    (lldb) f 4
    frame #4: 0x000000010357d7c6 libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x000000011b4938c0, mm=0x000000010ad3f7a0) + 1558 at OGeo.cc:502
       499  
       500      NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
       501      NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
    -> 502      NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
       503      NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim
       504      NPY<unsigned>*  idBuf = mm->getAnalyticInstancedIdentityBuffer(); assert(idBuf && ( idBuf->hasShape(-1,4) || idBuf->hasShape(-1,1,4)));
       505       // PmtInBox yielding -1,1,4 ?
    (lldb) p planBuf
    (NPY<float> *) $2 = 0x0000000000000000
    (lldb) p partBuf
    (NPY<float> *) $3 = 0x000000010ad410c0
    (lldb) p tranBuf
    (NPY<float> *) $4 = 0x000000010ad412c0
    (lldb) p primBuf
    (NPY<int> *) $5 = 0x00007fff5fbfbe20
    (lldb) 


