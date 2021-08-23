identity_review
==================


IdentityTests
--------------

GGeoIdentityTest
    for all merged meshes loop over all volumes accessing identity 


Changes
---------

1. non-detector specific sensor interface in G4Opticks::setSensorData providing a way 
   to get the sensor identifier into Opticks with no detector specific assumptions
2. added OpticksIdentity encoding triplet ridx/pidx/oidx into every GNode in the geometry tree


Questions
-----------

1. how/where to use the sensor data  



Where does instancedIdentity come from ?
--------------------------------------------

::

    864 void GInstancer::makeMergedMeshAndInstancedBuffers(unsigned verbosity)
    865 {
    866     bool last = false ;
    867 
    868     //const GNode* root = m_nodelib->getNode(0);
    869     assert(m_root);
    870     GNode* base = NULL ;
    871 
    872 
    873     // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    874     unsigned ridx0 = 0 ;
    875     GMergedMesh* mm0 = m_geolib->makeMergedMesh(ridx0, base, m_root);
    876 
    877 
    878     std::vector<const GNode*> placements = getPlacements(ridx0);  // just m_root
    879     assert(placements.size() == 1 );
    880     mm0->addInstancedBuffers(placements);  // call for global for common structure 
    881 
    882     unsigned numRepeats = getNumRepeats();
    883     unsigned numRidx = 1 + numRepeats ;
    884 
    885     LOG(LEVEL) << " numRepeats " << numRepeats << " numRidx " << numRidx ;
    886 
    887     for(unsigned ridx=1 ; ridx < numRidx ; ridx++)  // 1-based index
    888     {
    889          const GNode*   rbase  = last ? getLastRepeatExample(ridx)  : getRepeatExample(ridx) ;
    890 
    891          LOG(LEVEL) << " ridx " << ridx << " rbase " << rbase ;
    892 
    893          GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, rbase, m_root );
    894 
    895          std::vector<const GNode*> placements_ = getPlacements(ridx);
    896 
    897          mm->addInstancedBuffers(placements_);
    898 
    899          std::string dmesh = mm->descMeshUsage( m_ggeo, "GInstancer::makeMergedMeshAndInstancedBuffers.descMeshUsage" );
    900          LOG(LEVEL) << dmesh ;
    901 
    902          std::string dskip = mm->descBoundarySkip(m_ggeo, "GInstancer::makeMergedMeshAndInstancedBuffers.descBoundarySkip" );
    903          LOG(LEVEL) << dskip ;
    904     }
    905 
    906 
    907 }

    1245 /**
    1246 GMergedMesh::addInstancedBuffers
    1247 -----------------------------------
    1248 
    1249 itransforms InstanceTransformsBuffer
    1250     (num_instances, 4, 4)
    1251 
    1252     collect GNode placement transforms into buffer
    1253 
    1254 iidentity InstanceIdentityBuffer
    1255     From Aug 2020: (num_instances, num_volumes_per_instance, 4 )
    1256     Before:        (num_instances*num_volumes_per_instance, 4 )
    1257 
    1258     collects the results of GVolume::getIdentity for all volumes within all instances. 
    1259 
    1260 **/
    1261 
    1262 void GMergedMesh::addInstancedBuffers(const std::vector<const GNode*>& placements)
    1263 {
    1264     LOG(LEVEL) << " placements.size() " << placements.size() ;
    1265 
    1266     NPY<float>* itransforms = GTree::makeInstanceTransformsBuffer(placements);
    1267     setITransformsBuffer(itransforms);
    1268 
    1269     NPY<unsigned int>* iidentity  = GTree::makeInstanceIdentityBuffer(placements);
    1270     setInstancedIdentityBuffer(iidentity);
    1271 }
    1272 



TODO: Identity Packing
------------------------

::

     369 /**
     370 G4Opticks::setSensorData
     371 ---------------------------
     372 
     373 Calls to this for all sensor_placements G4PVPlacement provided by G4Opticks::getSensorPlacements
     374 provides a way to associate the Opticks contiguous 0-based sensorIndex with a detector 
     375 defined sensor identifier. 
     376 
     377 Within JUNO simulation framework this is used from LSExpDetectorConstruction::SetupOpticks.
     378 
     379 
     380 sensorIndex 
     381     0-based continguous index used to access the sensor data, 
     382     the index must be less than the number of sensors
     383 efficiency_1 
     384 efficiency_2
     385     two efficiencies which are multiplied together with the local angle dependent efficiency 
     386     to yield the detection efficiency used to assign SURFACE_COLLECT to photon hits 
     387     that already have SURFACE_DETECT 
     388 category
     389     used to distinguish between sensors with different theta textures   
     390 identifier
     391     detector specific integer representing a sensor, does not need to be contiguous
     392 
     393 
     394 Within JUNO simulation framework this is used from LSExpDetectorConstruction::SetupOpticks
     395 whilst looping over the sensor_placements G4PVPlacement provided by G4Opticks::getSensorPlacements.
     396 
     397 **/
     398 
     399 void G4Opticks::setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int category, int identifier)
     400 {   
     401     assert( sensorIndex < m_sensor_num ); 
     402     m_sensor_data->setFloat(sensorIndex,0,0,0, efficiency_1);
     403     m_sensor_data->setFloat(sensorIndex,1,0,0, efficiency_2);
     404     m_sensor_data->setInt(  sensorIndex,2,0,0, category);
     405     m_sensor_data->setInt(  sensorIndex,3,0,0, identifier);
     406 }


::

    245 /**
    246 GVolume::getIdentity
    247 ----------------------
    248 
    249 The volume identity quad is available GPU side for all intersects
    250 with geometry.
    251 
    252 1. node_index (3 bytes at least as JUNO needs more than 2-bytes : so little to gain from packing) 
    253 2. triplet_identity (4 bytes, pre-packed)
    254 3. SPack::Encode22(mesh_index, boundary_index)
    255 
    256    * mesh_index: 2 bytes easily enough, 0xffff = 65535
    257    * boundary_index: 2 bytes easily enough  
    258 
    259 4. sensor_index (2 bytes easily enough) 
    260 
    261 The sensor_identifier is detector specific so would have to allow 4-bytes 
    262 hence exclude it from this identity, instead can use sensor_index to 
    263 look up sensor_identifier within G4Opticks::getHit 
    264 
    265 Formerly::
    266 
    267    guint4 id(getIndex(), getMeshIndex(),  getBoundary(), getSensorIndex()) ;
    268 
    269 **/
    270 guint4 GVolume::getIdentity() const
    271 {
    272     guint4 id(getIndex(), getTripletIdentity(),  getShapeIdentity(), getSensorIndex()) ;
    273     return id ; 
    274 }   
    275 glm::uvec4 GVolume::getIdentity_() const
    276 {
    277     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    278     return id ; 
    279 }   
    280 
    281 /**
    282 GVolumne::getShapeIdentity
    283 ----------------------------
    284 
    285 The shape identity packs mesh index and boundary index together.
    286 This info is used GPU side by::
    287 
    288    oxrap/cu/material1_propagate.cu:closest_hit_propagate
    289 
    290 **/
    291 
    292 unsigned GVolume::getShapeIdentity() const
    293 {
    294     return SPack::Encode22( getMeshIndex(), getBoundary() );
    295 }   
    296 




users of identity.z instanceIdentity.z
----------------------------------------

::

     52 RT_PROGRAM void closest_hit_propagate()
     53 {
     54      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     55      float cos_theta = dot(n,ray.direction);
     56 
     57      prd.cos_theta = cos_theta ;
     58      prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this
     59 
     60      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     61      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     62      prd.identity = instanceIdentity ;
     63      prd.surface_normal = cos_theta > 0.f ? -n : n 



sensor_index or sensor_identifier provided from GVolume::getIdentity
-----------------------------------------------------------------------

* use the opticks sensor index, which as contiguous, can be contained in 2 bytes (0xffff = 65535)
* also the sensor_data needs to be copied to GPU anyhow for the efficiencies, so can do sensor_index keyed 
  lookups both on GPU and CPU as needed

where the sensor info is used
--------------------------------

* due to the detect property of the surface get some SURFACE_DETECT flag, which results 
  in the hits being copied back to CPU 

oxrap/cu/generate.cu::

    631         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    632         {
    633             command = propagate_at_surface(p, s, rng);
    634             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    635             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    636         }
    637         else
    638         {
    639             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    640             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    641             // tacit CONTINUE
    642         }


Photon flags.u.y hold identity.w oxrap/cu/generate.cu::

    213 #define FLAGS(p, s, prd) \
    214 { \
    215     p.flags.i.x = prd.boundary ;  \
    216     p.flags.u.y = s.identity.w ;  \
    217     p.flags.u.w |= s.flag ; \
    218 } \
    ...


Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc::


    540     int merged_count(0);
    541     for(int i=0 ; i < nhit ; i++)
    542     {
    543         g4ok->getHit(i,
    544                      &position,
    545                      &time,
    546                      &direction,
    547                      &weight,
    548                      &polarization,
    549                      &wavelength,
    550                      &flags_x,
    551                      &flags_y,
    552                      &flags_z,
    553                      &flags_w,
    554                      &is_cerenkov,
    555                      &is_reemission
    556                     );
    557 
    558         int pmtid = flags_y ;

    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    using sensor_index means need to do a lookup for the hits to get the 
    detector specific sensor identifier

    TODO: 
       getHit should provide the sensor_identifier given by G4Opticks::setSensorData
       rather than raw flags  


    559         G4double hittime = time ;
    560 
    561         bool merged = false ;
    562         if (m_pmthitmerger_opticks and m_pmthitmerger_opticks->getMergeFlag()) {
    563             merged = m_pmthitmerger_opticks->doMerge(pmtid, hittime);
    564         }





X4PhysicalVolume::convertNode tracing back where the sensor info comes from
------------------------------------------------------------------------------

::


    1207 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1208 {
    ...
    1213     // record copynumber in GVolume, as thats one way to handle pmtid
    1214     const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    1215     assert(placement);
    1216     G4int copyNumber = placement->GetCopyNo() ;
    ...
    1220     unsigned boundary = addBoundary( pv, pv_p );
    1221     std::string boundaryName = m_blib->shortname(boundary);
    1222     int materialIdx = m_blib->getInnerMaterial(boundary);
    ...
    1366     int sensorIndex = m_blib->isSensorBoundary(boundary) ? m_ggeo->addSensorVolume(volume) : -1 ;
    1367     if(sensorIndex > -1) m_blib->countSensorBoundary(boundary);
    ...
    1385     volume->setSensorIndex(sensorIndex);


    1046 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
    1047 {
    1048     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    1049     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
    1050 
    1051     const G4Material* const imat_ = lv->GetMaterial() ;
    1052     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
    1053 


    0529 bool GBndLib::isSensorBoundary(unsigned boundary) const
     530 {
     531     const guint4& bnd = m_bnd[boundary];
     532     bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]);
     533     bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]);
     534     bool is_sensor = osur_sensor || isur_sensor ;
     535     return is_sensor ;
     536 }

    898 // m_sensor_indices is a transient (non-persisted) vector of material/surface indices 
    899 bool GPropertyLib::isSensorIndex(unsigned index) const
    900 {
    901     typedef std::vector<unsigned>::const_iterator UI ;
    902     UI b = m_sensor_indices.begin();
    903     UI e = m_sensor_indices.end();
    904     UI i = std::find(b, e, index);
    905     return i != e ;
    906 }


    908 /**
    909 GPropertyLib::addSensorIndex
    910 ------------------------------
    911 
    912 Canonically invoked from GSurfaceLib::collectSensorIndices
    913 
    914 **/
    915 void GPropertyLib::addSensorIndex(unsigned index)
    916 {
    917     m_sensor_indices.push_back(index);
    918 }


    0288 template <class T>
     289 bool GPropertyMap<T>::isSensor()
     290 {
     291 #ifdef OLD_SENSOR
     292     return m_sensor ;
     293 #else
     294     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     295 #endif
     296 }

    0723 /**
     724 GSurfaceLib::collectSensorIndices
     725 ----------------------------------
     726 
     727 Loops over all surfaces collecting the 
     728 indices of surfaces having non-zero EFFICIENCY or detect
     729 properties.
     730 
     731 **/
     732 
     733 void GSurfaceLib::collectSensorIndices()
     734 {
     735     unsigned ni = getNumSurfaces();
     736     for(unsigned i=0 ; i < ni ; i++)
     737     {
     738         GPropertyMap<float>* surf = m_surfaces[i] ;
     739         bool is_sensor = surf->isSensor() ; 
     740         if(is_sensor)
     741         {
     742             addSensorIndex(i);
     743             assert( isSensorIndex(i) == true ) ;
     744         }   
     745     }   
     746 }   






TODO: getting the user input sensor_identifier onto the GNode tree 
--------------------------------------------------------------------

* G4Opticks::getSensorArray 



GPU side access to identity 
----------------------------

Three flavors of access to identity:

1. GeometryTriangles : the new form of RTX acceleration triangle intersection introduced with OptiX 6.0
2. TriangleMesh : old familiar triangle mesh 
3. Analytic : directly InstanceIdentityBuffer with identity at volume level 

Triangulated identity duplicates the volume level according to the number of triangles for each volume,
such that every triangle gets the identity.


identityBuffer sources depend on geocode of the GMergedMesh
-------------------------------------------------------------

OGeo::makeGeometryTriangles
     GBuffer* rib = mm->getAppropriateRepeatedIdentityBuffer() ;

OGeo::makeTriangulatedGeometry
     GBuffer* id = mm->getAppropriateRepeatedIdentityBuffer();

OGeo::makeAnalyticGeometry
     NPY<unsigned>*  idBuf = mm->getInstancedIdentityBuffer();


What is Appropriate
--------------------

::

    2242 /**
    2243 GMesh::getAppropriateRepeatedIdentityBuffer
    2244 ---------------------------------------------
    2245 
    2246 mmidx > 0 (FORMERLY: numITransforms > 0)
    2247    friib : FaceRepeatedInstancedIdentityBuffer 
    2248 
    2249 frib (FORMERLY: numITransforms == 0)
    2250    frib :  FaceRepeatedIdentityBuffer
    2251 
    2252 
    2253 Sep 2020: moved to branching on mmidx > 0 as that 
    2254 matches the rest of the geometry conversion code.  
    2255 In anycase numITransforms is never zero. 
    2256 For global mmidx=0 it is always 1 (identity matrix). 
    2257 So was previously always returning friib.
    2258 
    2259 **/
    2260 
    2261 GBuffer*  GMesh::getAppropriateRepeatedIdentityBuffer()
    2262 {
    2263     GMesh* mm = this ;
    2264     unsigned numITransforms = mm->getNumITransforms();
    2265     unsigned numFaces = mm->getNumFaces();
    2266     unsigned mmidx = mm->getIndex(); 
    2267     
    2268     GBuffer* id = NULL ;
    2269     
    2270     if(mmidx > 0)
    2271     {
    2272         id = mm->getFaceRepeatedInstancedIdentityBuffer();
    2273         assert(id);
    2274         LOG(LEVEL) << "using FaceRepeatedInstancedIdentityBuffer" << " friid items " << id->getNumItems() << " numITransforms*numFaces " << numITransforms*numFaces ;
    2275         assert( id->getNumItems() == numITransforms*numFaces );
    2276     }   
    2277     else
    2278     {
    2279         id = mm->getFaceRepeatedIdentityBuffer();
    2280         assert(id);
    2281         LOG(LEVEL) << "using FaceRepeatedIdentityBuffer" << " frid items " << id->getNumItems() << " numFaces " << numFaces ;
    2282         assert( id->getNumItems() == numFaces );
    2283     }   
    2284     return id ;
    2285 }   
    2286 




use of the identity within the GPU geometry intersect code
------------------------------------------------------------

::

    epsilon:cu blyth$ grep identityBuffer *.*


    GeometryTriangles.cu:rtBuffer<uint4>  identityBuffer; 
    GeometryTriangles.cu:    const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced

    TriangleMesh.cu:rtBuffer<uint4>  identityBuffer; 
    TriangleMesh.cu:    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced

    csg_intersect_boolean.h:            instanceIdentity = identityBuffer[instance_index*primitive_count+primIdx] ;
    intersect_analytic.cu:identityBuffer sources depend on geocode of the GMergedMesh
    intersect_analytic.cu:rtBuffer<uint4>  identityBuffer;   
    intersect_analytic.cu:    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ; 
    intersect_analytic.cu:    uint4 identity_test = identityBuffer[instance_index_test*primitive_count+primIdx] ; 
    intersect_analytic.cu:identityBuffer
    sphere.cu:rtBuffer<uint4>  identityBuffer; 
    sphere.cu:  uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced



Notice that there are separate identityBuffer for each of the GMergedMesh (mm), access is via:: 

     uint4 identity = identityBuffer[instance_index*primitive_count+primIdx]
   
instance_index 
   over all the instances, NB for global this is zero 
primIdx
   index over the primitive_count volumes within each instance



IDEA 1 : combined getIdentity getInstancedIdentity ?
-----------------------------------------------------

The identity info is the same, the difference between these is the indexing.

::


     552 /**
     553 GMesh::getInstancedIdentity
     554 -----------------------------
     555 
     556 All nodes of the geometry tree have a quad of identity uint.
     557 InstancedIdentity exists to rearrange that identity information 
     558 into a buffer that can be used for creation of the GPU instanced geometry,
     559 which requires to access the identity with an instance index, rather 
     560 than the node index.
     561 
     562 See notes/issues/identity_review.rst
     563 
     564 **/
     565 
     566 guint4 GMesh::getInstancedIdentity(unsigned int index) const
     567 {
     568     return m_iidentity[index] ;
     569 }
     570 


::

    1180 /**
    1181 GMergedMesh::addInstancedBuffers
    1182 -----------------------------------
    1183 
    1184 itransforms InstanceTransformsBuffer
    1185     (num_instances, 4, 4)
    1186 
    1187     collect GNode placement transforms into buffer
    1188 
    1189 iidentity InstanceIdentityBuffer
    1190     From Aug 2020: (num_instances, num_volumes_per_instance, 4 )
    1191     Before:        (num_instances*num_volumes_per_instance, 4 )
    1192 
    1193     collects the results of GVolume::getIdentity for all volumes within all instances. 
    1194 
    1195 **/
    1196 
    1197 void GMergedMesh::addInstancedBuffers(const std::vector<GNode*>& placements)
    1198 {
    1199     LOG(LEVEL) << " placements.size() " << placements.size() ;
    1200 
    1201     NPY<float>* itransforms = GTree::makeInstanceTransformsBuffer(placements);
    1202     setITransformsBuffer(itransforms);
    1203 
    1204     NPY<unsigned int>* iidentity  = GTree::makeInstanceIdentityBuffer(placements);
    1205     setInstancedIdentityBuffer(iidentity);
    1206 }




QUDARap identity
------------------

Old school identity : using identityBuffer
-------------------------------------------------

* see notes/issues/identity_review.rst 


oxrap/cu/intersect_analytic.cu::

    094 rtDeclareVariable(unsigned int, instance_index,  ,);
     95 // optix::GeometryInstance instance_index into the identity buffer, 
     96 // set by oxrap/OGeo.cc, 0 for non-instanced 
     97 
     98 rtDeclareVariable(unsigned int, primitive_count, ,);
     99 rtDeclareVariable(unsigned int, repeat_index, ,);
    100 
    101 
    102 rtBuffer<Part> partBuffer;
    103 
    104 rtBuffer<Matrix4x4> tranBuffer;
    105 
    106 rtBuffer<Prim>  primBuffer;
    107 
    108 rtBuffer<uint4>  identityBuffer;
        

oxrap/cu/csg_intersect_boolean.h::

    0707 static __device__
     708 void evaluative_csg( const Prim& prim, const int primIdx )   // primIdx just used for identity access
     709 {
    ...
    1023         if(rtPotentialIntersection( fabsf(ret.w) ))
    1024         {
    1025             shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
    1026             instanceIdentity = identityBuffer[instance_index*primitive_count+primIdx] ;

    // NB: THIS IS FROM A GEOMETRY MODEL SPLIT ON GMergedMesh 
    //
    // HMM : THIS IS FINE FOR THE INSTANCES BECAUSE NOT MANY PRIMITIVES EACH 
    // BUT TRYING TO HANDLE GLOBAL INTERSECTS TOO LIKE THIS WOULD LEAD TO MOSTLY EMPTY identityBuffer 

oxrap/cu/closest_hit_propagate.cu::

     26 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
     27      
     28 rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
     29 rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
     30 rtDeclareVariable(float,                  t, rtIntersectionDistance, );
     31 
     32 RT_PROGRAM void closest_hit_propagate()
     33 {    
     34      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     35      float cos_theta = dot(n,ray.direction);
     36      
     37      prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it
     38 
     39      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
     40      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     41      prd.identity = instanceIdentity ; 
     42      prd.surface_normal = cos_theta > 0.f ? -n : n ;   
     43 }

CSG/CSGFoundry identity
--------------------------

CSG/csg_intersect_node.h::

    1091 INTERSECT_FUNC
    1092 bool intersect_prim( float4& isect, int numNode, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
    1093 {
    1094     return numNode == 1
    1095                ?
    1096                   intersect_node(isect,          node, plan, itra, t_min, ray_origin, ray_direction )
    1097                :
    1098                   intersect_tree(isect, numNode, node, plan, itra, t_min, ray_origin, ray_direction )
    1099                ;
    1100 }

* CSGPrim is either one CSGNode or a sequence of them
* the CSG model has greater focus on the CSGNode : so can see why identity/boundary info should be held there 
  even if repeated across all CSGNode of a CSGPrim

* is a separate identity buffer needed ? or is there enough space to keep inside the CSGNode 
* hmm maybe it cannot be kept there because its not "mesh" level, its "structure" level 
  because of G4LogicalBorderSurface depending on pv pairs   


Could q3.u.w hold the boundary::  

    166     QAT4_METHOD void getIdentity(unsigned& ins_idx, unsigned& gas_idx, unsigned& ias_idx ) const
    167     {
    168         ins_idx = q0.u.w - 1u ;
    169         gas_idx = q1.u.w - 1u ;
    170         ias_idx = q2.u.w - 1u ;
    171     }
    172     QAT4_METHOD void setIdentity(unsigned ins_idx, unsigned gas_idx, unsigned ias_idx )
    173     {
    174         q0.u.w = ins_idx + 1u ;
    175         q1.u.w = gas_idx + 1u ;
    176         q2.u.w = ias_idx + 1u ;
    177     }




CSGFoundry Intersect Identity : missing *boundary*
---------------------------------------------------


CSGNode::setBoundary not used yet, hmm why not boundary on CSGPrim ?

* because the CSG model focus is on CSGNode

::


    088 struct CSGNode
    089 {
    ...
    154     NODE_METHOD unsigned boundary()  const {      return q1.u.z ; }
    155     NODE_METHOD void setBoundary(unsigned bnd){          q1.u.z = bnd ; }


* need to review intersect code 



GGeo -> CSGFoundry : minimal identity handling
---------------------------------------------------------

::

    140 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
    141 {   
    142     unsigned nmm = ggeo->getNumMergedMesh();
    143     assert( repeatIdx < nmm ); 
    144     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
    145     unsigned num_inst = mm->getNumITransforms() ;
    146     
    147     //LOG(LEVEL) << " nmm " << nmm << " repeatIdx " << repeatIdx << " num_inst " << num_inst ; 
    148     
    149     for(unsigned i=0 ; i < num_inst ; i++)
    150     {   
    151         glm::mat4 it = mm->getITransform_(i);
    152         qat4 instance(glm::value_ptr(it)) ;   
    153         unsigned ins_idx = foundry->inst.size() ;
    154         unsigned gas_idx = repeatIdx ;
    155         unsigned ias_idx = 0 ; 
    156         instance.setIdentity( ins_idx, gas_idx, ias_idx );
    ///    TODO: retain the "i" here so can backtrack
    157         foundry->inst.push_back( instance );
    158     }
    159 }


Need to find a place for the iid in the CSGFoundry model::

    2021-08-23 12:00:06.632 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 0 iid 1,3084,4
    2021-08-23 12:00:06.632 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 1 iid 25600,5,4
    2021-08-23 12:00:06.635 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 2 iid 12612,3,4
    2021-08-23 12:00:06.638 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 3 iid 5000,3,4
    2021-08-23 12:00:06.638 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 4 iid 2400,4,4
    2021-08-23 12:00:06.639 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 5 iid 590,1,4
    2021-08-23 12:00:06.639 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 6 iid 590,1,4
    2021-08-23 12:00:06.639 INFO  [1753424] [CSG_GGeo_Convert::addInstances@148]  reapeatIdx 7 iid 590,1,4

Now to handle identity info with such different shapes in a collective way ?

* flatten into (n, 4) and keep the flat index in CSGFoundry model 
* but there is currently nowhere to keep that flat index (only have the inst)
* could consolidate into (n,4) for upload and calculate an array of pointers to give split access for each ridx
* keep split asis and work out how to do lookups

::

    xNP::Write dtype <i4 ni       10 nj  3 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name solid.npy
    xNP::Write dtype <f4 ni     3233 nj  4 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name prim.npy
    xNP::Write dtype <f4 ni    17619 nj  4 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name node.npy
    xNP::Write dtype <f4 ni     8184 nj  4 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name tran.npy
    xNP::Write dtype <f4 ni     8184 nj  4 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name itra.npy
    xNP::Write dtype <f4 ni    48477 nj  4 nk  4 nl  -1 nm  -1 dir             /tmp/blyth/opticks/CSG_GGeo/CSGFoundry name inst.npy

* exactly what identity info is needed on GPU  : boundary for sure
* is mesh-level vs structure-level for boundary really needed ? OR can the info be placed on the prim/node ?
 
Current simple identity.::

    331     unsigned instance_idx = optixGetInstanceId() ;    // see IAS_Builder::Build and InstanceId.h 
    332     unsigned prim_idx  = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    333     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_idx & 0xffff ) ;

* flat instance_idx could via the inst give repeatIdx and the split GMergedMesh instance index "i" above, then prim_idx gives the 2nd index  
* so can get back to the uint4 identity CPU side easily : BUT what is needed GPU side ? 

  * CAN the boundary live in the CSGPrim/CSGNode ? 

  * boundary index is defined as a unique set of four indices (omat, osur, isur, imat) 
    that summarizes surface/material information of a geometry during a volume traverse that 
    looks at parent/child G4VPhysicalVolume and G4LogicalVolume (see X4PhysicalVolume::convertNode/X4PhysicalVolume::addBoundary)

  * although border surfaces depend on PV pairs making them a bit more structural(PV level) than mesh/shape(LV) level, 
    the reality of usage is that things like boundary do not vary by instance : so should 
    be able to plant them on the CSGPrim/CSGNode 
 
    * need a way to check this ?

  * one thing that always does vary by instance is the sensor identifier and sensor efficiencies 




