identity_review
==================


IdentityTests
--------------

GGeoIdentityTest
    for all merged meshes loop over all volumes accessing identity 


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





