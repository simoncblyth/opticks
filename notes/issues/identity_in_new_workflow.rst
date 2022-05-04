identity_in_new_workflow
===========================

overview
----------

* per-specific-PMT info like efficiencies must be at held at instance level, 
  so cannot be CSGNode info which is at LogicalVolume/shape level

* PMT type could be obtained from boundary howver 


Current state of new workflow
---------------------------------

* NB the iid (instanced identity) quad is not currently being placed into the CSGFoundry geom


TODO: below or simular can be used on GMergedMesh to get instanced identity::  

     623 guint4 GMesh::getInstancedIdentity(unsigned int index) const
     624 {
     625     return m_iidentity[index] ;
     626 }
     627 
     628 glm::uvec4 GMesh::getInstancedIdentity_(unsigned index) const
     629 {
     630     guint4 id = m_iidentity[index];
     631     return id.as_vec();
     632 }

::

    0208 /**
     209 CSG_GGeo_Convert::addInstances
     210 ---------------------------------
     211 
     212 Invoked from tail of CSG_GGeo_Convert::convertSolid which 
     213 gets called for all repeatIdx, including global repeatIdx 0 
     214 
     215 Notice the flattening effect, this is consolidating the 
     216 transforms from all GMergedMesh into one foundry->inst vector of qat4.
     217 
     218 TODO: put the iid somewhere 
     219 
     220 **/
     221 
     222 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     223 {
     224     unsigned nmm = ggeo->getNumMergedMesh();
     225     assert( repeatIdx < nmm );
     226     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
     227     unsigned num_inst = mm->getNumITransforms() ;
     228     NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();
     229 
     230     LOG(LEVEL)
     231         << " repeatIdx " << repeatIdx
     232         << " num_inst (GMergedMesh::getNumITransforms) " << num_inst
     233         << " iid " << ( iid ? iid->getShapeString() : "-"  )
     234         ;
     235 
     236     //LOG(LEVEL) << " nmm " << nmm << " repeatIdx " << repeatIdx << " num_inst " << num_inst ; 
     237 
     238     for(unsigned i=0 ; i < num_inst ; i++)
     239     {
     240         glm::mat4 it = mm->getITransform_(i);
     241 
     242         const float* tr16 = glm::value_ptr(it) ;
     243         unsigned gas_idx = repeatIdx ;
     244         unsigned ias_idx = 0 ;
     245 
     246         foundry->addInstance(tr16, gas_idx, ias_idx);
     247     }
     248 }


    1363 /**
    1364 CSGFoundry::addInstance
    1365 ------------------------
    1366 
    1367 Used for example from CSG_GGeo_Convert::addInstances
    1368 
    1369 **/
    1370 
    1371 void CSGFoundry::addInstance(const float* tr16, unsigned gas_idx, unsigned ias_idx )
    1372 {
    1373     qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    1374     unsigned ins_idx = inst.size() ;
    1375 
    1376     instance.setIdentity( ins_idx, gas_idx, ias_idx );
    1377 
    1378     LOG(LEVEL)
    1379         << " ins_idx " << ins_idx
    1380         << " gas_idx " << gas_idx
    1381         << " ias_idx " << ias_idx
    1382         ;
    1383 
    1384     inst.push_back( instance );
    1385 }

Only 12 of the 16 values of 4x4 transform actually needed.  
So use those 4 spares to carry identity info inside the instance transform.::

    302     QAT4_METHOD void setIdentity(unsigned ins_idx, unsigned gas_idx, unsigned ias_idx )
    303     {
    304         q0.u.w = ins_idx + 1u ;
    305         q1.u.w = gas_idx + 1u ;
    306         q2.u.w = ias_idx + 1u ;
    307     }



GGeo level : which is a part of both old and new workflows
-----------------------------------------------------------------

::

    226 /**
    227 GVolume::getIdentity
    228 ----------------------
    229 
    230 The volume identity quad is available GPU side for all intersects
    231 with geometry.
    232 
    233 1. node_index (3 bytes at least as JUNO needs more than 2-bytes : so little to gain from packing) 
    234 2. triplet_identity (4 bytes, pre-packed)
    235 3. SPack::Encode22(mesh_index, boundary_index)
    236 
    237    * mesh_index: 2 bytes easily enough, 0xffff = 65535
    238    * boundary_index: 2 bytes easily enough  
    239 
    240 4. sensorIndex (2 bytes easily enough) 
    241 
    242 The sensor_identifier is detector specific so would have to allow 4-bytes 
    243 hence exclude it from this identity, instead can use sensorIndex to 
    244 look up sensor_identifier within G4Opticks::getHit 
    245 
    246 Formerly::
    247 
    248    guint4 id(getIndex(), getMeshIndex(),  getBoundary(), getSensorIndex()) ;
    249 
    250 **/
    251 
    252 glm::uvec4 GVolume::getIdentity() const
    253 {
    254     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    255     return id ;
    256 }


    088 /**
     89 GTree::makeInstanceIdentityBuffer : (numPlacements, numVolumes, 4 )
     90 ----------------------------------------------------------------------
     91 
     92 Canonically invoked by GMergedMesh::addInstancedBuffers
     93 
     94 Collects identity quads from the GVolume(GNode) tree into an array, 
     95 
     96 Repeating identity guint4 for all volumes of an instance (typically ~5 volumes for 1 instance)
     97 into all the instances (typically large 500-36k).
     98 
     99 Instances need to know the sensor they correspond to 
    100 even though their geometry is duplicated. 
    101 
    102 For analytic geometry this is needed at the volume level 
    103 ie need buffer of size: num_transforms * num_volumes-per-instance
    104 
    105 For triangulated geometry this is needed at the triangle level
    106 ie need buffer of size: num_transforms * num_triangles-per-instance
    107 
    108 The triangulated version can be created from the analytic one
    109 by duplication according to the number of triangles.
    110 
    111 Prior to Aug 2020 this returned an iidentity buffer with all nodes 
    112 when invoked on the root node, eg::  
    113 
    114     GMergedMesh/0/iidentity.npy :       (1, 316326, 4)
    115 
    116 This was because of a fundamental difference between the repeated instances and the 
    117 remainder ridx 0 volumes. The volumes of the instances are all together in subtrees 
    118 whereas the remainder volumes with ridx 0 are scattered all over the full tree. Thus 
    119 the former used of this with GGeo::m_root as the only placement resulted in getting 
    120 base + progeny covering all nodes of the tree. To avoid this a separate getRemainderProgeny 
    121 is now used which selects the collected nodes based on the ridx (getRepeatIndex()) 
    122 being zero.
    123 
    124 **/



::

    1265 void GMergedMesh::addInstancedBuffers(const std::vector<const GNode*>& placements)
    1266 {
    1267     LOG(LEVEL) << " placements.size() " << placements.size() ;
    1268 
    1269     NPY<float>* itransforms = GTree::makeInstanceTransformsBuffer(placements);
    1270     setITransformsBuffer(itransforms);
    1271 
    1272     NPY<unsigned int>* iidentity  = GTree::makeInstanceIdentityBuffer(placements);
    1273     setInstancedIdentityBuffer(iidentity);
    1274 }




old workflow
--------------

oxrap/OGeo.cc populates separate identityBuffer for each MM (compound solid)::

     756 optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)
     757 {
     ...
     800     NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
     801     NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
     802     NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
     803     NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim
     804 
     805     // NB these buffers are concatenations of the corresponding buffers for multiple prim 
     806     unsigned numPrim = primBuf->getNumItems();
     807 
     808     NPY<float>* itransforms = mm->getITransformsBuffer(); assert(itransforms && itransforms->hasShape(-1,4,4) ) ;
     809     unsigned numInstances = itransforms->getNumItems();
     ...
     810     NPY<unsigned>*  idBuf = mm->getInstancedIdentityBuffer();   assert(idBuf);
     ...
     897 
     898     geometry["primitive_count"]->setUint( numPrim );       // needed GPU side, for instanced offset into buffers 
     899     geometry["repeat_index"]->setUint( mm->getIndex() );  // ridx
     900     geometry["analytic_version"]->setUint(analytic_version);
     901 
     902     optix::Program intersectProg = m_ocontext->createProgram("intersect_analytic.cu", "intersect") ;
     903     optix::Program boundsProg  =  m_ocontext->createProgram("intersect_analytic.cu", "bounds") ;
     904 
     905     geometry->setIntersectionProgram(intersectProg );
     906     geometry->setBoundingBoxProgram( boundsProg );
     ...
     921     optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer");
     922     geometry["identityBuffer"]->setBuffer(identityBuffer);

ocu/intersect_analytic.cu::

    098 rtDeclareVariable(unsigned int, primitive_count, ,);
    ...
    108 rtBuffer<uint4>  identityBuffer;
    ...
    167 rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    168 rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
    169 rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
    ...
    176 #include "csg_intersect_boolean.h"


ocu/csg_intersect_boolean.h sets instanceIdentity ::

    0707 static __device__
     708 void evaluative_csg( const Prim& prim, const int primIdx )   // primIdx just used for identity access
     709 {
    ...
    1023         if(rtPotentialIntersection( fabsf(ret.w) ))
    1024         {
    1025             shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
    1026             instanceIdentity = identityBuffer[instance_index*primitive_count+primIdx] ;
    1027 
    1028 #ifdef BOOLEAN_DEBUG
    1029             instanceIdentity.x = ierr > 0 ? 1 : 0 ;   // used for visualization coloring  
    1030             instanceIdentity.y = ierr ;
    1031             // instanceIdentity.z is used for boundary passing, hijacking prevents photon visualization
    1032             instanceIdentity.w = tloop ;
    1033 #endif
    1034 
    1035 //#define WITH_PRINT_IDENTITY_INTERSECT_TAIL 1 
    1036 #ifdef WITH_PRINT_IDENTITY_INTERSECT_TAIL
    1037             rtPrintf("// csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index %d instance_index %d primitive_count %3d primIdx %3d instanceIdentity ( %7d %7d      %7d %7d )   \n",
    1038             repeat_index, instance_index, primitive_count, primIdx, instanceIdentity.x, instanceIdentity.y, instanceIdentity.z, instanceIdentity.w  );
    1039 #endif
    1040 
    1041             rtReportIntersection(0);
    1042         }




