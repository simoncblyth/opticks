GGeo-direct-review
====================

Review the use of GGeo by the direct geometry workflow.  The direct 
workflow means via g4ok:G4Opticks/x4:X4PhysicalVolume.


Direct Geometry Workflow kicked off by G4Opticks::Initialize
---------------------------------------------------------------

::

    114 void G4Opticks::Initialize(const char* gdmlpath, bool standardize_geant4_materials)
    115 {
    116     const G4VPhysicalVolume* world = CGDML::Parse(gdmlpath);
    117     Initialize(world, standardize_geant4_materials);
    118 }
    119 
    120 void G4Opticks::Initialize(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
    121 {
    122     G4Opticks* g4ok = GetOpticks();
    123     g4ok->setGeometry(world, standardize_geant4_materials) ;
    124 }


    238 void G4Opticks::setGeometry(const G4VPhysicalVolume* world, bool standardize_geant4_materials)
    239 {
    ...
    244     GGeo* ggeo = translateGeometry( world ) ;


    292 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    293 {
    ...
    323     GGeo* gg = new GGeo(ok, live) ;
    324     LOG(info) << ") GGeo instanciate " ;
    325 
    326     LOG(info) << "( GGeo populate" ;
    327     X4PhysicalVolume xtop(gg, top) ;
    328     LOG(info) << ") GGeo populate" ;
    329 
    330     LOG(info) << "( GGeo::postDirectTranslation " ;
    331     gg->postDirectTranslation();
    332     LOG(info) << ") GGeo::postDirectTranslation " ;



Translation Steered from X4PhysicalVolume
--------------------------------------------

::

    0179 void X4PhysicalVolume::init()
     180 {
     181     LOG(LEVEL) << "[" ; 
     182     LOG(LEVEL) << " query : " << m_query->desc() ;
     183 
     184 
     185     convertMaterials();   // populate GMaterialLib
     186     convertSurfaces();    // populate GSurfaceLib
     187     // convertSensors();  // before closeSurfaces as may add some SensorSurfaces
     188     closeSurfaces();
     189     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVolume)  
     190     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     191     convertCheck();
     192     
     193     LOG(LEVEL) << "]" ;
     194 }   


     710 GMesh* X4PhysicalVolume::convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname, bool balance_deep_tree ) const
     711 {
     712      assert( lvIdx == soIdx );
     ...        
     724      nnode* raw = X4Solid::Convert(solid, m_ok)  ;
     ...
     732      NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
     ...
     745      GMesh* mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid ) ;
     746      mesh->setCSG( csg ); 
     ...
     749      return mesh ; 
     750 }



    1160 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1161 {
    ...
    1177     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    1180     unsigned ndIdx = m_node_count ;       // incremented below after GVolume instanciation
    1181 
    1182     int lvIdx = m_lvidx[lv] ;   // from postorder traverse in convertSolids to match GDML lvIdx : mesh identity uses lvIdx
    ...
    1195     const GMesh* mesh = m_hlib->getMeshWithIndex(lvIdx); 
    1196 
    1197     const NCSG* csg = mesh->getCSG();  
    1198     unsigned csgIdx = csg->getIndex() ; 
    ...
    1305     GVolume* volume = new GVolume(ndIdx, gtransform, mesh );
    1306     m_node_count += 1 ; 
    1307 
    ...
    1316     NSensor* sensor = NULL ; 
    1317     volume->setSensor( sensor );   
    1318     volume->setCopyNumber(copyNumber); 
    1319     volume->setBoundary( boundary ); 
    1320     volume->setSelected( selected );
    1321 
    1322     volume->setLevelTransform(ltransform);
    1323 
    1324     volume->setLocalTransform(ltriple);
    1325     volume->setGlobalTransform(gtriple);
    ....
    1340     if(parent) 
    1341     {
    1342          parent->addChild(volume);
    1343          volume->setParent(parent);
    1344     } 
    ....
    1353     return volume ; 
    1354 }




GGeo details
-----------------


::

    0793 void GGeo::postDirectTranslation()
     794 {   
     797     prepare();
     ...
     805     save();
     809 }

     822 void GGeo::prepare()
     823 {
     832     prepareScintillatorLib();
     835     prepareSourceLib();
     838     prepareVolumes();   // GInstancer::createInstancedMergedMeshes
     841     prepareVertexColors();  // writes colors into GMergedMesh mm0
     844 }



    1436 void GGeo::prepareVolumes()
    1437 {   
    1438     LOG(info) << "[ creating merged meshes from the volume tree " ;
    ....
    1454         bool deltacheck = true ;
    1455         m_instancer->createInstancedMergedMeshes(deltacheck, meshverbosity);   // GInstancer::createInstancedMergedMeshes
    ....
    1466 }   


    103 void GInstancer::createInstancedMergedMeshes(bool delta, unsigned verbosity)
    104 {
    111     traverse();   // spin over tree counting up progenyDigests to find repeated geometry 
    115     labelTree();  // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
    119     makeMergedMeshAndInstancedBuffers(verbosity);
    125 }


    673 void GInstancer::makeMergedMeshAndInstancedBuffers(unsigned verbosity)
    674 {
    677     GNode* root = m_nodelib->getNode(0);
    679     GNode* base = NULL ;
    ...
    682     // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    683     unsigned ridx0 = 0 ;
    684     GMergedMesh* mm0 = m_geolib->makeMergedMesh(ridx0, base, root, verbosity );
    686 
    687     std::vector<GNode*> placements = getPlacements(ridx0);  // just m_root
    688     assert(placements.size() == 1 );
    689     mm0->addInstancedBuffers(placements);  // call for global for common structure 
    ...
    700     for(unsigned ridx=1 ; ridx < numRidx ; ridx++)  // 1-based index
    701     {
    702          GNode*   rbase  = last ? getLastRepeatExample(ridx)  : getRepeatExample(ridx) ;
    710          GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, rbase, root, verbosity );
    712          std::vector<GNode*> placements_ = getPlacements(ridx);
    714          mm->addInstancedBuffers(placements_);
    717     }
    718 }


    302 GMergedMesh* GGeoLib::makeMergedMesh(unsigned index, GNode* base, GNode* root, unsigned verbosity )
    303 {
    306     if(m_merged_mesh.find(index) == m_merged_mesh.end())
    307     {
    308         m_merged_mesh[index] = GMergedMesh::Create(index, base, root, verbosity );
    309     }
    310     GMergedMesh* mm = m_merged_mesh[index] ;
    314     return mm ;
    315 }


    0238 GMergedMesh* GMergedMesh::Create(unsigned ridx, GNode* base, GNode* root, unsigned verbosity ) // static
     239 {
     240     assert(root && "root node is required");
     252     GMergedMesh* mm = new GMergedMesh( ridx );
     253     mm->setCurrentBase(base);  // <-- when NULL it means will use global not base relative transforms
     254 
     255     GNode* start = base ? base : root ;
     263     mm->traverse_r( start, 0, PASS_COUNT, verbosity  );  // 1st pass traversal : counts vertices and faces
     280     mm->allocate();  // allocate space for flattened arrays
     284     mm->traverse_r( start, 0, PASS_MERGE, verbosity );  // 2nd pass traversal : merge copy GMesh into GMergedMesh 
     288     mm->updateBounds();
     294     return mm ;
     295 }

     318 void GMergedMesh::traverse_r( GNode* node, unsigned depth, unsigned pass, unsigned verbosity )
     319 {
     320     GVolume* volume = dynamic_cast<GVolume*>(node) ;
     ...
     353     switch(pass)
     354     {
     355         case PASS_COUNT:    countVolume(volume, selected, verbosity)  ;break;
     356         case PASS_MERGE:    mergeVolume(volume, selected, verbosity)  ;break;
     357                 default:    assert(0)                                 ;break;
     358     }
     359 
     360     for(unsigned i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
     361 }


     364 void GMergedMesh::countVolume( GVolume* volume, bool selected, unsigned verbosity )
     365 {
     366     const GMesh* mesh = volume->getMesh();
     367     m_num_volumes += 1 ;
     369     if(selected)
     370     {
     371         m_num_volumes_selected += 1 ;
     372         countMesh( mesh );
     373     }
     382 }

     384 void GMergedMesh::countMesh( const GMesh* mesh )
     385 {
     386     unsigned nface = mesh->getNumFaces();
     387     unsigned nvert = mesh->getNumVertices();
     388     unsigned meshIndex = mesh->getIndex();
     389 
     390     m_num_vertices += nvert ;
     391     m_num_faces    += nface ;
     392     m_mesh_usage[meshIndex] += 1 ;  // which meshes contribute to the mergedmesh
     393 }



     482 void GMergedMesh::mergeVolume( GVolume* volume, bool selected, unsigned verbosity )
     483 {
     484     GNode* node = static_cast<GNode*>(volume);
     485     GNode* base = getCurrentBase();
     486     unsigned ridx = volume->getRepeatIndex() ;
     487 
     488     GMatrixF* transform = base ? volume->getRelativeTransform(base) : volume->getTransform() ;     // base or root relative global transform
     489 
     494     float* dest = getTransform(m_cur_volume);
     495     assert(dest);
     496     transform->copyTo(dest);
     497 
     498     const GMesh* mesh = volume->getMesh();   // triangulated
     499     unsigned num_vert = mesh->getNumVertices();
     500     unsigned num_face = mesh->getNumFaces();
     501 
     510     guint3* faces = mesh->getFaces();
     511     gfloat3* vertices = mesh->getTransformedVertices(*transform) ;
     512     gfloat3* normals  = mesh->getTransformedNormals(*transform);
     513 
     515     mergeVolumeBBox(vertices, num_vert);
     516     mergeVolumeIdentity(volume, selected );
     517 
     518     m_cur_volume += 1 ;    // irrespective of selection, as prefer absolute volume indexing 
     519 
     520     if(selected)
     521     {
     523         mergeVolumeVertices( num_vert, vertices, normals );
     525         unsigned* node_indices     = volume->getNodeIndices();
     526         unsigned* boundary_indices = volume->getBoundaryIndices();
     527         unsigned* sensor_indices   = volume->getSensorIndices();
     529         mergeVolumeFaces( num_face, faces, node_indices, boundary_indices, sensor_indices  );
     536         GPt* pt = volume->getPt();  // analytic 
     537         mergeVolumeAnalytic( pt, transform, verbosity );
     540         // offsets with the flat arrays
     541         m_cur_vertices += num_vert ;
     542         m_cur_faces    += num_face ;
     543     }
     544 }


     700 void GMergedMesh::mergeVolumeIdentity( GVolume* volume, bool selected )
     701 {
     702     const GMesh* mesh = volume->getMesh();
     703 
     704     unsigned nvert = mesh->getNumVertices();
     705     unsigned nface = mesh->getNumFaces();
     706 
     707     guint4 _identity = volume->getIdentity();
     708 
     709     unsigned nodeIndex = volume->getIndex();
     710     unsigned meshIndex = mesh->getIndex();
     711     unsigned boundary = volume->getBoundary();
     712 
     713     NSensor* sensor = volume->getSensor();
     714     unsigned sensorIndex = NSensor::RefIndex(sensor) ;
     715 
     716     assert(_identity.x == nodeIndex);
     717     assert(_identity.y == meshIndex);
     718     assert(_identity.z == boundary);
     719     //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
     720    
     730     GNode* parent = volume->getParent();
     731     unsigned int parentIndex = parent ? parent->getIndex() : UINT_MAX ;
     732 
     733     m_meshes[m_cur_volume] = meshIndex ;

     738     m_nodeinfo[m_cur_volume].x = selected ? nface : 0 ;
     739     m_nodeinfo[m_cur_volume].y = selected ? nvert : 0 ;
     740     m_nodeinfo[m_cur_volume].z = nodeIndex ;
     741     m_nodeinfo[m_cur_volume].w = parentIndex ;
     ...
     753     m_identity[m_cur_volume] = _identity ;
     754 }



Problem with m_copyNumber/identity_index(=pmtID) is that its not an index, it has great big gaps::

    202 guint4 GVolume::getIdentity()
    203 {
    204     unsigned node_index = m_index ;
    205 
    206     //unsigned identity_index = getSensorSurfaceIndex() ;   
    207     unsigned identity_index = m_copyNumber  ;
    208 
    209     // surprised to get this in the global 
    210     //if(identity_index > 300000 ) std::raise(SIGINT); 
    211 
    212     return guint4(
    213                    node_index,
    214                    getMeshIndex(),
    215                    m_boundary,
    216                    identity_index
    217                  );
    218 }


Placement transforms collected into m_pts(GPts) are relative to the instance base. 
Note one GPt added for each volume within the instance subtree::

     859 void GMergedMesh::mergeVolumeAnalytic( GPt* pt, GMatrixF* transform, unsigned /*verbosity*/ )
     860 {
     863     const float* data = static_cast<float*>(transform->getPointer());
     865     glm::mat4 placement = glm::make_mat4( data ) ;
     867     pt->setPlacement(placement);
     869     m_pts->add( pt );
     870 }   


x4/X4PhysicalVolume.cc::

    1216     GPt* pt = new GPt( lvIdx, ndIdx, csgIdx, boundaryName.c_str() )  ;
    1333     volume->setPt( pt );


::

    epsilon:GPts blyth$ inp ?/iptBuffer.npy -l
    a :                                              0/iptBuffer.npy :             (374, 4) : cfcb7b3c1f2314b02ed20609f687c52f : 20200719-2129 
    b :                                              1/iptBuffer.npy :               (5, 4) : 0a7c1e906a6a3913f3bcfe3ab4d40dd7 : 20200719-2129 
    c :                                              2/iptBuffer.npy :               (6, 4) : 42761fa2b500a8fd70d9f67416f9c916 : 20200719-2129 
    d :                                              3/iptBuffer.npy :               (6, 4) : a7d635662dee3dc1ea006fd36a18763f : 20200719-2129 
    e :                                              4/iptBuffer.npy :               (6, 4) : d0650e08593ea37ed79aab92cab13604 : 20200719-2129 
    f :                                              5/iptBuffer.npy :               (1, 4) : 547da34217547f78916d7ec9f136ed9a : 20200719-2129 
    g :                                              6/iptBuffer.npy :               (1, 4) : d26bda9e14e82bf4a256d1098084e692 : 20200719-2129 
    h :                                              7/iptBuffer.npy :               (1, 4) : 07fdae2d906fed39fedc7e95ca7136d5 : 20200719-2129 
    i :                                              8/iptBuffer.npy :               (1, 4) : 2ff7c7568240328b81716a99ab93f5ef : 20200719-2129 
    j :                                              9/iptBuffer.npy :             (130, 4) : 8c925a62dc2af568e967e927da9b52b5 : 20200719-2129 

    In [1]: d   (6,4) (num_volumes, num_qty)
    Out[1]: 
    array([[   35, 68256,    35,     0],
           [   30, 68257,    30,     1],
           [   34, 68258,    34,     2],
           [   33, 68259,    33,     3],
           [   31, 68260,    31,     4],
           [   32, 68261,    32,     5]], dtype=int32)

          ## lvIdx ndIdx  csgIdx             csgIdx 31 and 32 are the ones with the problem 




Postcache deferred formation of the analytic GParts geometry, using the persistable GPts(m_pts) from each GMergedMesh::

    0827 void OpticksHub::deferredGeometryPrep()
     828 {   
     829     m_ggeo->deferredCreateGParts() ;
     830 }

    1484 void GGeo::deferredCreateGParts()
    1485 {
    1488     const std::vector<const NCSG*>& solids = m_meshlib->getSolids();
    1490     unsigned verbosity = 0 ;
    1492     unsigned nmm = m_geolib->getNumMergedMesh();
    1499     for(unsigned i=0 ; i < nmm ; i++)
    1500     {
    1501         GMergedMesh* mm = m_geolib->getMergedMesh(i);
    1512         GPts* pts = mm->getPts();
    1519         GParts* parts = GParts::Create( pts, solids, verbosity ) ;
    1520         parts->setBndLib(m_bndlib);
    1521         parts->close();
    1523         mm->setParts( parts );
    1524     }
    1527 }


    0191 The (GPt)pt from each GVolume yields a per-volume (GParts)parts instance
     192 that is added to the (GParts)com instance.

    0212 GParts* GParts::Create(const GPts* pts, const std::vector<const NCSG*>& solids, unsigned verbosity) // static
     213 {
     216     GParts* com = new GParts() ;
     218     unsigned num_pt = pts->getNumPt();
     222     for(unsigned i=0 ; i < num_pt ; i++)
     223     {
     224         const GPt* pt = pts->getPt(i);
     225         int   lvIdx = pt->lvIdx ;
     226         int   ndIdx = pt->ndIdx ;
     227         const std::string& spec = pt->spec ;
     228         const glm::mat4& placement = pt->placement ;
     231         const NCSG* csg = unsigned(lvIdx) < solids.size() ? solids[lvIdx] : NULL ;
     235         GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx );
     240         parts->applyPlacementTransform( placement );
     242         com->add( parts, verbosity );
     243     }
     245     return com ; 
     246 }





* separate optix::GeometryGroup for global or optix::Group for the instances 
  with separate intersect program contexts 

::

     280 void OGeo::convertMergedMesh(unsigned i)
     281 {
     282     LOG(LEVEL) << "( " << i  ;
     283     m_mmidx = i ;
     284 
     285     GMergedMesh* mm = m_geolib->getMergedMesh(i);
     286 
     287     bool raylod = m_ok->isRayLOD() ;
     288     if(raylod) LOG(fatal) << " RayLOD enabled " ;
     289 
     290     bool is_null = mm == NULL ;
     291     bool is_skip = mm->isSkip() ;
     292     bool is_empty = mm->isEmpty() ;
     293 
     294     if( is_null || is_skip || is_empty )
     295     {
     296         LOG(error) << " not converting mesh " << i << " is_null " << is_null << " is_skip " << is_skip << " is_empty " << is_empty ;
     297         return  ;
     298     }
     299 
     300     unsigned numInstances = 0 ;
     301     if( i == 0 )   // global non-instanced geometry in slot 0
     302     {
     303         optix::GeometryGroup ggg = makeGlobalGeometryGroup(mm);
     304         m_top->addChild(ggg);
     305         numInstances = 1 ;
     306     }
     307     else           // repeated geometry
     308     {
     309         optix::Group assembly = makeRepeatedAssembly(mm, raylod) ;
     310         assembly->setAcceleration( makeAcceleration(m_assembly_accel, false) );
     311         numInstances = assembly->getChildCount() ;
     312         m_top->addChild(assembly);
     313     }
     314     LOG(LEVEL) << ") " << i << " numInstances " << numInstances ;
     315 }



All the volumes of an instance handled in sequence of *primitive_count* prim::

     645 optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm, unsigned lod)
     646 {
     662     GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");
     664     if(pts->getPrimBuffer() == NULL)
     665     {   
     667         pts->close(); 
     669     }
     691     NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
     692     NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
     693     NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
     694     NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim
     696     NPY<unsigned>*  idBuf = mm->getAnalyticInstancedIdentityBuffer(); assert(idBuf && ( idBuf->hasShape(-1,4) || idBuf->hasShape(-1,1,4)));
     698 
     699     unsigned numPrim = primBuf->getNumItems();
     700     unsigned numPart = partBuf->getNumItems();
     701     unsigned numTran = tranBuf->getNumItems();
     702     unsigned numPlan = planBuf->getNumItems();
     703 
     704     unsigned numVolumes = mm->getNumVolumes();
     705     unsigned numVolumesSelected = mm->getNumVolumesSelected();
     706 
     742     optix::Geometry geometry = m_context->createGeometry();
     756     geometry->setPrimitiveCount( lod > 0 ? 1 : numPrim );  // lazy lod, dont change buffers, just ignore all but the 1st prim for lod > 0
     757 
     758     geometry["primitive_count"]->setUint( numPrim );       // needed GPU side, for instanced offset into buffers 
     759     geometry["analytic_version"]->setUint(analytic_version);
     760 
     761     optix::Program intersectProg = m_ocontext->createProgram("intersect_analytic.cu", "intersect") ;
     762     optix::Program boundsProg  =  m_ocontext->createProgram("intersect_analytic.cu", "bounds") ;
     763 
     764     geometry->setIntersectionProgram(intersectProg );
     765     geometry->setBoundingBoxProgram( boundsProg );
     766 
     768     optix::Buffer primBuffer = createInputUserBuffer<int>( primBuf,  4*4, "primBuffer");
     769     geometry["primBuffer"]->setBuffer(primBuffer);
     771 
     773     optix::Buffer partBuffer = createInputUserBuffer<float>( partBuf,  4*4*4, "partBuffer");
     774     geometry["partBuffer"]->setBuffer(partBuffer);
     775 
     777     optix::Buffer tranBuffer = createInputUserBuffer<float>( tranBuf,  sizeof(optix::Matrix4x4), "tranBuffer");
     778     geometry["tranBuffer"]->setBuffer(tranBuffer);
     779 
     780     optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer");
     781     geometry["identityBuffer"]->setBuffer(identityBuffer);
     782 
     783     optix::Buffer planBuffer = createInputUserBuffer<float>( planBuf,  4*4, "planBuffer");
     784     geometry["planBuffer"]->setBuffer(planBuffer);
     785 
     787     optix::Buffer prismBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
     788     prismBuffer->setFormat(RT_FORMAT_FLOAT4);
     789     prismBuffer->setSize(5);
     790     geometry["prismBuffer"]->setBuffer(prismBuffer);
     791 
     799     return geometry ;
     800 }


::

    287 RT_PROGRAM void intersect(int primIdx)
    288 {
    289     const Prim& prim    = primBuffer[primIdx];
    290 
    291     unsigned partOffset  = prim.partOffset() ;
    292     unsigned numParts    = prim.numParts() ;
    293     unsigned primFlag    = prim.primFlag() ;
    294 
    295     uint4 identity = identityBuffer[instance_index] ;
    ^^^^^^^^^^^^^ aii buffer ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        For per-prim identity with ii buffer 

             uint4 identity = identityBuffer[instance_index*primitive_count+primIdx]  



    296 
    297     if(primFlag == CSG_FLAGNODETREE)
    298     {
    299         Part pt0 = partBuffer[partOffset + 0] ;
    300 
    301         identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary
    302 
    303         evaluative_csg( prim, identity );
    304         //intersect_csg( prim, identity );
    305 


      



::

    epsilon:GMergedMesh blyth$ np.py ?/*iidentity.npy 
    a :                                             0/aiidentity.npy :            (1, 1, 4) : 554f351ec53ee0d0e126796a23301a48 : 20200702-2350 
    b :                                              0/iidentity.npy :          (316326, 4) : 2a0515dd3da7723f1e6430ecb14536fa : 20200702-2350 
    c :                                             1/aiidentity.npy :        (25600, 1, 4) : 2656f9e5f92a858ac5c3d931bf4859fe : 20200702-2350 
    d :                                              1/iidentity.npy :          (128000, 4) : 925e98ab591dcdde40a42777b8331e9d : 20200702-2350 
    e :                                             2/aiidentity.npy :        (12612, 1, 4) : 930501c72265943bd9664c699196ff4e : 20200702-2350 
    f :                                              2/iidentity.npy :           (75672, 4) : a01fcebdd01b8c02fe22115fe43ef7c9 : 20200702-2350 
    g :                                             3/aiidentity.npy :         (5000, 1, 4) : 7f35c4d8c4c3ba493006bc67a4d065b3 : 20200702-2350 
    h :                                              3/iidentity.npy :           (30000, 4) : e9c45b8853360f9aaba32c363364925c : 20200702-2350 
    i :                                             4/aiidentity.npy :         (2400, 1, 4) : 46b03571ecbbc13524ee66609527258e : 20200702-2350 
    j :                                              4/iidentity.npy :           (14400, 4) : 98531585d60875dbe406e8552ded3306 : 20200702-2350 
    k :                                             5/aiidentity.npy :          (590, 1, 4) : c50545c210af57623423220dc03669f9 : 20200702-2350 
    l :                                              5/iidentity.npy :             (590, 4) : f6dbd22f73291140613bcd2caa376173 : 20200702-2350 
    m :                                             6/aiidentity.npy :          (590, 1, 4) : 0204aafef48e6e8908c2b9d4c00b37cf : 20200702-2350 
    n :                                              6/iidentity.npy :             (590, 4) : 78526fb5a6c83d01a40a92b293b36ad1 : 20200702-2350 
    o :                                             7/aiidentity.npy :          (590, 1, 4) : b29d7b61b3a690da481ba809b07dfd22 : 20200702-2350 
    p :                                              7/iidentity.npy :             (590, 4) : ea82e0dec5285b6ecd94a5fd151d3b47 : 20200702-2350 
    q :                                             8/aiidentity.npy :          (590, 1, 4) : b6d773513d04dc9910c26600b96aab1b : 20200702-2350 
    r :                                              8/iidentity.npy :             (590, 4) : 622c3f8e2b757113fd5534bba7ded8b7 : 20200702-2350 
    s :                                             9/aiidentity.npy :          (504, 1, 4) : 511f4f79cf6efb4d5358a7824f0ddf68 : 20200702-2350 
    t :                                              9/iidentity.npy :           (65520, 4) : 6bf8691f386e4c3c5645dd45699319c5 : 20200702-2350 
    epsilon:GMergedMesh blyth$ 




instanceIdentity : connection between the geometry intersect and the closest hit
----------------------------------------------------------------------------------

::

     27 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
     ..
     52 RT_PROGRAM void closest_hit_propagate()
     53 {
     54      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     55      float cos_theta = dot(n,ray.direction);
     56 
     57      prd.cos_theta = cos_theta ;
     58      prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this
     59      unsigned int boundaryIndex = instanceIdentity.z ;
     60      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     61      prd.identity = instanceIdentity ;
     62      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     63 }


intersect_analytic.cu::

    084 // attributes communicate to closest hit program,
     85 // they must be set inbetween rtPotentialIntersection and rtReportIntersection
     86 
     87 rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
     88 rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
     89 rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
     90 


    epsilon:cu blyth$ grep -l rtPotentialIntersection *.*
    GeometryTriangles.cu
    TriangleMesh.cu
    csg_intersect_boolean.h
    intersect_analytic.cu
    intersect_box.h
    intersect_prism.h
    intersect_zsphere.h
    intersect_ztubs.h
    sphere.cu



intersect_analytic.cu::

    287 RT_PROGRAM void intersect(int primIdx)
    288 {
    289     const Prim& prim    = primBuffer[primIdx];
    290 
    291     unsigned partOffset  = prim.partOffset() ;
    292     unsigned numParts    = prim.numParts() ;
    293     unsigned primFlag    = prim.primFlag() ;
    294 
    295     uint4 identity = identityBuffer[instance_index] ;
    296 
    297     if(primFlag == CSG_FLAGNODETREE)
    298     {
    299         Part pt0 = partBuffer[partOffset + 0] ;
    300 
    301         identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary
    302 
    303         evaluative_csg( prim, identity );
    304         //intersect_csg( prim, identity );
    305 
    306     }

What happens for global mm (ridx=0) ? with instance_index 0  
--------------------------------------------------------------------

Clearly should be using::

    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;

And switch from aii to ii ? 



Passing identity : intersect->closest_hit->raygen
---------------------------------------------------------------

csg_intersect_boolean.h::

     563 static __device__
     564 void evaluative_csg( const Prim& prim, const uint4& identity )
     565 {
     ...
     871     if(csg.curr == 0)
     872     {
     873          const float4& ret = csg.data[0] ;
     ...
     882          if(rtPotentialIntersection( fabsf(ret.w) ))
     883          {
     884               shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
     885               instanceIdentity = identity ;
     886 #ifdef BOOLEAN_DEBUG
     887               instanceIdentity.x = ierr > 0 ? 1 : 0 ;   // used for visualization coloring  
     888               instanceIdentity.y = ierr ;
     889               // instanceIdentity.z is used for boundary passing, hijacking prevents photon visualization
     890               instanceIdentity.w = tloop ;
     891 #endif
     892               rtReportIntersection(0);
     893          }
     894     }
     ...
     928 }



geocache identity
-------------------

::

    epsilon:~ blyth$ cd $GC
    epsilon:1 blyth$ t inp
    inp () 
    { 
        ipython -i $(which np.py) -- $*
    }

    [blyth@localhost 1]$ t inp
    inp is a function
    inp () 
    { 
        ~/anaconda2/bin/ipython -i -- ~/opticks/bin/np.py $*
    }



::

    [blyth@localhost 1]$ inp GMergedMesh/1/*.npy
    ...
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200720-2331 
    b :                                GMergedMesh/1/itransforms.npy :        (25600, 4, 4) : 1292187d260497fc0a694e275f0e6999 : 20200720-2331 
    c :                                    GMergedMesh/1/indices.npy :            (4752, 1) : b5d5dc7ce94690319fb384b1e503e2f9 : 20200720-2331 
    d :                                 GMergedMesh/1/boundaries.npy :            (1584, 1) : 4583b9e4b2524fc02d90306a4ae93238 : 20200720-2331 
    e :                                      GMergedMesh/1/nodes.npy :            (1584, 1) : 8cb9bf708067a07977010b6bc92bf565 : 20200720-2331 
    f :                                    GMergedMesh/1/sensors.npy :            (1584, 1) : 30e007064ccb81e841e90dde1304ccf2 : 20200720-2331 
    g :                                     GMergedMesh/1/colors.npy :             (805, 3) : 5b2f1391f85c6e29560eed612a0e890a : 20200720-2331 
    h :                                    GMergedMesh/1/normals.npy :             (805, 3) : 5482a46493c73523fdc5356fd6ed5ebc : 20200720-2331 
    i :                                   GMergedMesh/1/vertices.npy :             (805, 3) : b447acf665678da2789103b44874d6bb : 20200720-2331 
    j :                                       GMergedMesh/1/bbox.npy :               (5, 6) : a523db9c1220c034d29d8c0113b4ac10 : 20200720-2331 
    k :                              GMergedMesh/1/center_extent.npy :               (5, 4) : 3417b940f4da6db67abcf29937b52128 : 20200720-2331 
    l :                                   GMergedMesh/1/identity.npy :               (5, 4) : a921a71d379336f28e7c0b908eea9218 : 20200720-2331 
    m :                                     GMergedMesh/1/meshes.npy :               (5, 1) : 0a52a5397e61677ded7cd8a7b23bf090 : 20200720-2331 
    n :                                   GMergedMesh/1/nodeinfo.npy :               (5, 4) : c143e214851e70197a6de58b2c86b5a9 : 20200720-2331 
    o :                                 GMergedMesh/1/transforms.npy :              (5, 16) : 37ae1f7f4da2409596627cebfa5cb28b : 20200720-2331 

    In [1]: 

    [blyth@localhost 1]$ inp GMergedMesh/2/*.npy
    a :                                  GMergedMesh/2/iidentity.npy :        (12612, 6, 4) : 4423ba6434c39aff488e6784df468ae1 : 20200720-2331 
    b :                                GMergedMesh/2/itransforms.npy :        (12612, 4, 4) : c0ec856e88eb5ccdae839f25ab9c993e : 20200720-2331 
    c :                                    GMergedMesh/2/indices.npy :           (10800, 1) : ec2e48dfe19d0b2bbb6714b5d102ff1a : 20200720-2331 
    d :                                 GMergedMesh/2/boundaries.npy :            (3600, 1) : 7b4b60a99006ce8a5ca2668a9698c49e : 20200720-2331 
    e :                                      GMergedMesh/2/nodes.npy :            (3600, 1) : ad1b23ff95465e42e1ce0be6113b397b : 20200720-2331 
    f :                                    GMergedMesh/2/sensors.npy :            (3600, 1) : c09b7af09b553b5304da5a1559ca2c7d : 20200720-2331 
    g :                                     GMergedMesh/2/colors.npy :            (1820, 3) : 89ea4c93126cd1c14e27af2e499af434 : 20200720-2331 
    h :                                    GMergedMesh/2/normals.npy :            (1820, 3) : 0eb006545f4b8f605e0281d87b52f257 : 20200720-2331 
    i :                                   GMergedMesh/2/vertices.npy :            (1820, 3) : b8ea611275ec809336112591abcaa4a4 : 20200720-2331 
    j :                                       GMergedMesh/2/bbox.npy :               (6, 6) : 86926ee14d0e44cb937c9d4a87fe305f : 20200720-2331 
    k :                              GMergedMesh/2/center_extent.npy :               (6, 4) : d14e9c7b653990cfbfe2385653fbf22a : 20200720-2331 
    l :                                   GMergedMesh/2/identity.npy :               (6, 4) : 7c7a2c4bfb25e67c852aeac7d281c4f3 : 20200720-2331 
    m :                                     GMergedMesh/2/meshes.npy :               (6, 1) : 4ad6dd25bda1e2e9499f267a545aa75d : 20200720-2331 
    n :                                   GMergedMesh/2/nodeinfo.npy :               (6, 4) : 6f043e521cb6e9974fc3ab52a983c407 : 20200720-2331 
    o :                                 GMergedMesh/2/transforms.npy :              (6, 16) : a3570ab9415c863e270b39926702568a : 20200720-2331 

    In [1]: ii = a

    In [5]: ii.shape
    Out[5]: (12612, 6, 4)   # (num_instances, num_volumes, num_qty )

    In [3]: ii   
    Out[3]: 
    array([[[ 68250,     29,     21,      0],
            [ 68251,     24,     15,      0],
            [ 68252,     28,     22,      0],
            [ 68253,     27,     23,      0],
            [ 68254,     25,     24,      0],
            [ 68255,     26,     25,      0]],

           [[ 68262,     29,     21,      2],
            [ 68263,     24,     15,      2],
            [ 68264,     28,     22,      2],
            [ 68265,     27,     23,      2],
            [ 68266,     25,     24,      2],
            [ 68267,     26,     25,      2]],
           ...,

           [[173904,     29,     21,  17609],
            [173905,     24,     15,  17609],
            [173906,     28,     22,  17609],
            [173907,     27,     23,  17609],
            [173908,     25,     24,  17609],
            [173909,     26,     25,  17609]],

           [[173916,     29,     21,  17611],
            [173917,     24,     15,  17611],
            [173918,     28,     22,  17611],
            [173919,     27,     23,  17611],
            [173920,     25,     24,  17611],
            [173921,     26,     25,  17611]]], dtype=uint32)
 
         ## nodeIdx  meshIdx  bndIdx  copyNo/pmtid


::

    202 guint4 GVolume::getIdentity()
    203 {
    204     unsigned node_index = m_index ;
    205 
    206     //unsigned identity_index = getSensorSurfaceIndex() ;   
    207     unsigned identity_index = m_copyNumber  ;
    208 
    209     // surprised to get this in the global 
    210     //if(identity_index > 300000 ) std::raise(SIGINT); 
    211 
    212     return guint4(
    213                    node_index,
    214                    getMeshIndex(),
    215                    m_boundary,
    216                    identity_index
    217                  );
    218 }

::

    epsilon:1 blyth$  ~/opticks/bin/cat.py GItemList/GMeshLib.txt 29,24,28,27,25,26
    29   :1: NNVTMCPPMTsMask_virtual0x32a5060
    24   :1: NNVTMCPPMTsMask0x32a6070
    28   :1: NNVTMCPPMT_PMT_20inch_pmt_solid0x32a1b00
    27   :1: NNVTMCPPMT_PMT_20inch_body_solid0x32a2840
    25   :1: NNVTMCPPMT_PMT_20inch_inner1_solid0x32a3900
    26   :1: NNVTMCPPMT_PMT_20inch_inner2_solid0x32a3b70

    epsilon:1 blyth$ ~/opticks/ana/blib.py $PWD -s 21,15,22,23,24,25
     nbnd  35 nmat  16 nsur  20 
     21 : Water///Water           ## expected for the virtual mask "constainer"
     15 : Water///Acrylic         ## expected for the mask 
     22 : Water///Pyrex           ## OOPS : should this not be Acrylic///Pyrex ?
     23 : Pyrex///Pyrex           ## this is the crazy thin one 
     24 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     25 : Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 


    epsilon:1 blyth$ ~/opticks/bin/cat.py GItemList/GMeshLib.txt 35,30,34,33,31,32
    35   :1: HamamatsuR12860sMask_virtual0x3290560
    30   :1: HamamatsuR12860sMask0x3291550
    34   :2: HamamatsuR12860_PMT_20inch_pmt_solid_1_90x329ed30
    33   :2: HamamatsuR12860_PMT_20inch_body_solid_1_90x32b7d70
    31   :1: HamamatsuR12860_PMT_20inch_inner1_solid0x32a8f30
    32   :1: HamamatsuR12860_PMT_20inch_inner2_solid0x32a91b0

    epsilon:1 blyth$ ~/opticks/ana/blib.py $PWD -s 21,15,22,23,26,27
     nbnd  35 nmat  16 nsur  20 
     21 : Water///Water 
     15 : Water///Acrylic 
     22 : Water///Pyrex 
     23 : Pyrex///Pyrex 
     26 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     27 : Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum 


    epsilon:1 blyth$ ~/opticks/ana/blib.py $PWD -s 21,22,28,29,19
     nbnd  35 nmat  16 nsur  20 
     21 : Water///Water 
     22 : Water///Pyrex 
     28 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     29 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     19 : Water///Steel 

    epsilon:1 blyth$ ~/opticks/bin/cat.py GItemList/GMeshLib.txt 40,38,36,37,39
    40   :1: PMT_3inch_pmt_solid0x3a2d850
    38   :1: PMT_3inch_body_solid_ell_ell_helper0x3a2db10
    36   :1: PMT_3inch_inner1_solid_ell_helper0x3a2dba0
    37   :1: PMT_3inch_inner2_solid_ell_helper0x3a2dc80
    39   :1: PMT_3inch_cntr_solid0x3a2dd10




aii looking useless : have excluded it via WITH_AII
-----------------------------------------------------------

::


    In [4]: aii
    Out[4]: 
    array([[[ 68250,      0,      0,      0]],

           [[ 68262,      1,      0,      0]],

           [[ 68268,      2,      0,      0]],

           ...,

           [[173898,  12609,      0,      0]],

           [[173904,  12610,      0,      0]],

           [[173916,  12611,      0,      0]]], dtype=uint32)

    In [5]: 




Whats happening with global mm0 ?
------------------------------------

::

    2020-07-20 02:22:51.074 INFO  [339094] [OGeo::convert@264] [ nmm 10
    2020-07-20 02:22:51.074 INFO  [339094] [OGeo::convertMergedMesh@282] ( 0
    2020-07-20 02:22:51.076 INFO  [339094] [OGeo::makeOGeometry@590] ugeocode [A]
    2020-07-20 02:22:51.076 INFO  [339094] [OGeo::makeAnalyticGeometry@676]  skip GParts::close 
    2020-07-20 02:22:51.076 INFO  [339094] [OGeo::makeAnalyticGeometry@679] mm 0 verbosity: 0   pts:  GParts  primflag         flagnodetree numParts 1916 numPrim  374
    2020-07-20 02:22:51.076 INFO  [339094] [OGeo::makeAnalyticGeometry@709]  mmidx 0 numInstances 1 numPrim 374 idBuf 1,316326,4
    2020-07-20 02:22:51.076 FATAL [339094] [OGeo::makeAnalyticGeometry@738]  NodeTree : MISMATCH (numPrim != numVolumes)  (this happens when using --csgskiplv)  numVolumes 316326 numVolumesSelected 374 numPrim 374 numPart 1916 numTran 967 numPlan 0
    2020-07-20 02:22:51.379 INFO  [339094] [OGeo::convertMergedMesh@314] ) 0 numInstances 1
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::convertMergedMesh@282] ( 1
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeRepeatedAssembly@346]  mmidx 1 imodulo 0
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeRepeatedAssembly@366]  numTransforms 25600 numIdentity 25600 numSolids 1 islice NSlice      0 : 25600 :     1 
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeOGeometry@590] ugeocode [A]
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeAnalyticGeometry@676]  skip GParts::close 
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeAnalyticGeometry@679] mm 1 verbosity: 0   pts:  GParts  primflag         flagnodetree numParts    7 numPrim    5
    2020-07-20 02:22:51.380 INFO  [339094] [OGeo::makeAnalyticGeometry@709]  mmidx 1 numInstances 25600 numPrim 5 idBuf 25600,5,4
    2020-07-20 02:22:52.144 INFO  [339094] [OGeo::convertMergedMesh@314] ) 1 numInstances 25600




    2020-07-20 02:49:37.444 INFO  [395156] [OGeo::convert@264] [ nmm 10
    2020-07-20 02:49:37.444 INFO  [395156] [OGeo::convertMergedMesh@282] ( 0
    2020-07-20 02:49:37.446 INFO  [395156] [OGeo::makeOGeometry@590] ugeocode [T]
    2020-07-20 02:49:37.446 INFO  [395156] [OGeo::makeTriangulatedGeometry@926]  lod 0 mmIndex 0 numFaces (PrimitiveCount) 50136 numFaces0 (Outermost) 12 uFaces 50136 numVolumes 316326 numITransforms 1
    2020-07-20 02:49:37.446 INFO  [395156] [GMesh::makeFaceRepeatedInstancedIdentityBuffer@2115]  m_index 0 numITransforms 1 numVolumes 316326 numVolumesSelected 374 numFaces 50136 numRepeatedIdentity (numITransforms*numFaces) 50136 numInstanceIdentity 1
    2020-07-20 02:49:37.446 FATAL [395156] [GMesh::makeFaceRepeatedInstancedIdentityBuffer@2138] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 0 iidentity_buffer_items 1 numFaces (sum of faces in numVolumes)50136 numITransforms 1 numVolumes*numITransforms 316326 numRepeatedIdentity 50136
    python: /home/blyth/opticks/ggeo/GMesh.cc:2149: GBuffer* GMesh::makeFaceRepeatedInstancedIdentityBuffer(): Assertion `iidentity_ok' failed.

    Program received signal SIGABRT, Aborted.




mm0 idBuf 1,316326,4 is much too big, should be 1,374,4 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 316326 is the total number of nodes, not just the selected ridx=0 global ones 







    2020-07-21 01:04:35.297 INFO  [261343] [OContext::launch@786] PRELAUNCH time: 4.22139
    2020-07-21 01:04:35.297 INFO  [261343] [OPropagator::prelaunch@195] 0 : (0;0,0) 

::

    epsilon:1 blyth$ inp GMergedMesh/?/iidentity.npy 
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200719-2129 
    b :                                  GMergedMesh/2/iidentity.npy :        (12612, 6, 4) : 4423ba6434c39aff488e6784df468ae1 : 20200719-2129 
    c :                                  GMergedMesh/3/iidentity.npy :         (5000, 6, 4) : 52c59e1bb3179c404722c2df4c26ac81 : 20200719-2129 
    d :                                  GMergedMesh/4/iidentity.npy :         (2400, 6, 4) : 08846aa446e53c50c1a7cea89674a398 : 20200719-2129 
    e :                                  GMergedMesh/5/iidentity.npy :          (590, 1, 4) : 6b57bfe28d74e9e161a1a0908d568b84 : 20200719-2129 
    f :                                  GMergedMesh/6/iidentity.npy :          (590, 1, 4) : 45836c662ac5095c0d623bf7ed8a3399 : 20200719-2129 
    g :                                  GMergedMesh/7/iidentity.npy :          (590, 1, 4) : 92bdabddd8393af96cd10f43b8e920f2 : 20200719-2129 
    h :                                  GMergedMesh/8/iidentity.npy :          (590, 1, 4) : 98a9c18bdf1d64f1fa80a10799073b8d : 20200719-2129 
    i :                                  GMergedMesh/9/iidentity.npy :        (504, 130, 4) : 01278331416251ff7fd611fd2b1debd4 : 20200719-2129 
    j :                                  GMergedMesh/0/iidentity.npy :       (1, 316326, 4) : 57ddfde998a9f5ceab681b00b3b49e5b : 20200719-2129 


    In [4]: j[0]
    Out[4]: 
    array([[     0,     56,      0,      0],
           [     1,     12,      1,      0],
           [     2,     11,      2,      0],
           ...,
           [316323,     50,     23,  32399],
           [316324,     48,     33,  32399],
           [316325,     49,     34,  32399]], dtype=uint32)


    In [2]: a[0]
    Out[2]: 
    array([[173922,     40,     21, 300000],
           [173923,     38,     22, 300000],  primIdx 1 
           [173924,     36,     28, 300000],
           [173925,     37,     29, 300000],
           [173926,     39,     19, 300000]], dtype=uint32)

    In [7]: a[10]
    Out[7]: 
    array([[173972,     40,     21, 300010],
           [173973,     38,     22, 300010],
           [173974,     36,     28, 300010],
           [173975,     37,     29, 300010],
           [173976,     39,     19, 300010]], dtype=uint32)




    In [3]: b[0]
    Out[3]: 
    array([[68250,    29,    21,     0],
           [68251,    24,    15,     0],    primIdx 1
           [68252,    28,    22,     0],
           [68253,    27,    23,     0],
           [68254,    25,    24,     0],
           [68255,    26,    25,     0]], dtype=uint32)

    In [5]: c[0]
    Out[5]: 
    array([[68256,    35,    21,     1],
           [68257,    30,    15,     1],
           [68258,    34,    22,     1],
           [68259,    33,    23,     1],
           [68260,    31,    26,     1],
           [68261,    32,    27,     1]], dtype=uint32)



okt --printenabled pindex 1000 reveals issue
-----------------------------------------------

::

    2020-07-21 05:52:46.257 INFO  [240712] [OPropagator::resize@218]  m_oevt 0x2078a350 evt 0x581a4e0 numPhotons 11235 u_numPhotons 11235
    2020-07-21 05:52:46.257 INFO  [240712] [OPropagator::setSize@152]  width 11235 height 1
    2020-07-21 05:52:46.258 INFO  [240712] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-21 05:52:46.258 INFO  [240712] [OPropagator::launch@267] LAUNCH NOW -
    2020-07-21 05:52:46.258 INFO  [240712] [OContext::launch@783]  entry 0 width 11235 height 1  --printenabled  printLaunchIndex ( 1000 0 0)
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   0 identity (  161778      35      21   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   1 identity (  161779      30      15   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   2 identity (  161780      34      22   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   3 identity (  161781      33      23   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   4 identity (  161782      31      26   15588 ) 
    //evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   5 identity (  161783      32      27   15588 ) 
    //evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   0 identity (  161784      29      21   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   1 identity (  161785      24      15   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   2 identity (  161786      28      22   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   3 identity (  161787      27      23   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   4 identity (  161788      25      24   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 2 instance_index 11163 primitive_count   6 primIdx   5 identity (  161789      26      25   15589 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   0 identity (  161778      35      21   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   1 identity (  161779      30      15   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   2 identity (  161780      34      22   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   3 identity (  161781      33      23   15588 ) 
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   4 identity (  161782      31      26   15588 ) 
    //evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 4425 primitive_count   6 primIdx   5 identity (  161783      32      27   15588 ) 
    //evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit



::

    epsilon:1 blyth$ inp GMergedMesh/?/iidentity.npy 
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200719-2129 
    b :                                  GMergedMesh/2/iidentity.npy :        (12612, 6, 4) : 4423ba6434c39aff488e6784df468ae1 : 20200719-2129 
    c :                                  GMergedMesh/3/iidentity.npy :         (5000, 6, 4) : 52c59e1bb3179c404722c2df4c26ac81 : 20200719-2129 
    d :                                  GMergedMesh/4/iidentity.npy :         (2400, 6, 4) : 08846aa446e53c50c1a7cea89674a398 : 20200719-2129 
    e :                                  GMergedMesh/5/iidentity.npy :          (590, 1, 4) : 6b57bfe28d74e9e161a1a0908d568b84 : 20200719-2129 
    f :                                  GMergedMesh/6/iidentity.npy :          (590, 1, 4) : 45836c662ac5095c0d623bf7ed8a3399 : 20200719-2129 
    g :                                  GMergedMesh/7/iidentity.npy :          (590, 1, 4) : 92bdabddd8393af96cd10f43b8e920f2 : 20200719-2129 
    h :                                  GMergedMesh/8/iidentity.npy :          (590, 1, 4) : 98a9c18bdf1d64f1fa80a10799073b8d : 20200719-2129 
    i :                                  GMergedMesh/9/iidentity.npy :        (504, 130, 4) : 01278331416251ff7fd611fd2b1debd4 : 20200719-2129 
    j :                                  GMergedMesh/0/iidentity.npy :       (1, 316326, 4) : 57ddfde998a9f5ceab681b00b3b49e5b : 20200719-2129 



