seqvol : volume sequence indexing
===================================

Problem with volume sequencing is the large number of volumes and expensive storage of int32*10 sequence
but actually the number relevant to critical optical path is not so big, 
so judicious favoriting of 15 volumes 0x0->0xE specific to targetted AD and overflow 0xF for all others
may be sufficient.


closest_hit_propagate
------------------------

oxrap/cu/material1_propagate.cu::

     01 #include <optix.h>
      2 #include "PerRayData_propagate.h"
      3 #include "wavelength_lookup.h"
      4 
      5 //attributes set by TriangleMesh.cu:mesh_intersect 
      6 
      7 rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
      8 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
      9 
     10 rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
     11 rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
     12 rtDeclareVariable(float,                  t, rtIntersectionDistance, );
     13 
     14 
     15 RT_PROGRAM void closest_hit_propagate()
     16 {
     17      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     18 
     19      float cos_theta = dot(n,ray.direction);
     20 
     21      prd.cos_theta = cos_theta ;
     22 
     23      prd.distance_to_boundary = t ;
     24 
     25      unsigned int boundaryIndex = instanceIdentity.z ;
     26 
     27      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     28 
     29      prd.identity = instanceIdentity ;
     30 
     31      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     32 
     33 }


instance_identity comes from the intersects
---------------------------------------------


::

    delta:cu blyth$ grep instance_identity *.*
    TriangleMesh.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    hemi-pmt.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    material1_propagate.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    material1_radiance.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    sphere.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);


mesh_intersect
----------------

::

    010 // inputs from OGeo
     11 rtBuffer<int3>   indexBuffer;
     12 rtBuffer<float3> vertexBuffer;
     13 rtBuffer<uint4>  identityBuffer;
     14 rtDeclareVariable(unsigned int, instance_index,  ,);
     15 rtDeclareVariable(unsigned int, primitive_count, ,);
     16 
     17 // attribute variables communicating from intersection program to closest hit program
     18 // (must be set between rtPotentialIntersection and rtReportIntersection)
     19 rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
     20 rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
     21 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     22 
     23 
     24 
     25 RT_PROGRAM void mesh_intersect(int primIdx)
     26 {
     27     int3 index = indexBuffer[primIdx];
     28 
     29     float3 p0 = vertexBuffer[index.x];
     30     float3 p1 = vertexBuffer[index.y];
     31     float3 p2 = vertexBuffer[index.z];
     32 
     33     uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced
     34 
     35     float3 n;
     36     float  t, beta, gamma;
     37     if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
     38     {
     39         if(rtPotentialIntersection( t ))
     40         {
     41             geometricNormal = normalize(n);
     42             instanceIdentity = identity ;
     .. 
     53             rtReportIntersection(0);    // material index 0 
     54         }
     55     }
     56 }



oxrap/cu/hemi-pmt.cu::

    1248 RT_PROGRAM void intersect(int primIdx)
    1249 {
    1250   const uint4& solid    = solidBuffer[primIdx];
    1251   unsigned int numParts = solid.y ;
    1252 
    1253   //const uint4& identity = identityBuffer[primIdx] ; 
    1254   //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    1255 
    1256   // try with just one identity per-instance 
    1257   uint4 identity = identityBuffer[instance_index] ;
    1258 
    1259 
    1260   for(unsigned int p=0 ; p < numParts ; p++)
    1261   {
    1262       unsigned int partIdx = solid.x + p ;
    1263 
    1264       quad q0, q1, q2, q3 ;
    1265 
    1266       q0.f = partBuffer[4*partIdx+0];
    1267       q1.f = partBuffer[4*partIdx+1];
    1268       q2.f = partBuffer[4*partIdx+2] ;
    1269       q3.f = partBuffer[4*partIdx+3];
    1270 
    1271       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1272 
    1273       int partType = q2.i.w ;
    1274 
    1275       // TODO: use enum
    1276       switch(partType)
    1277       {
    1278           case 0:
    1279                 intersect_aabb(q2, q3, identity);
    1280                 break ;
    1281           case 1:
    1282                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1283                 break ;



identityBuffer
----------------

::

    delta:cfg4 blyth$ opticks-find identityBuffer
    ./optixrap/cu/hemi-pmt.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/hemi-pmt.cu:  uint4 identity = identityBuffer[instance_index] ; 
    ./optixrap/cu/hemi-pmt.cu:  //const uint4& identity = identityBuffer[primIdx] ; 
    ./optixrap/cu/hemi-pmt.cu:  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    ./optixrap/cu/hemi-pmt.cu:  uint4 identity = identityBuffer[instance_index] ; 
    ./optixrap/cu/sphere.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/sphere.cu:  uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    ./optixrap/cu/TriangleMesh.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/TriangleMesh.cu:    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced
    ./ggeo/GPmt.cc:    792   const uint4& identity = identityBuffer[primIdx] ;
    ./optixrap/OGeo.cc:    optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    ./optixrap/OGeo.cc:    geometry["identityBuffer"]->setBuffer(identityBuffer);
    ./optixrap/OGeo.cc:   optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    ./optixrap/OGeo.cc:   geometry["identityBuffer"]->setBuffer(identityBuffer);


OGeo.cc::

    537     optix::Geometry geometry = m_context->createGeometry();
    538     geometry->setIntersectionProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_intersect"));
    539     geometry->setBoundingBoxProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_bounds"));
    540 
    541     unsigned int numSolids = mm->getNumSolids();
    542     unsigned int numFaces = mm->getNumFaces();
    543     unsigned int numITransforms = mm->getNumITransforms();
    544 
    545     geometry->setPrimitiveCount(numFaces);
    546     assert(geometry->getPrimitiveCount() == numFaces);
    547     geometry["primitive_count"]->setUint( geometry->getPrimitiveCount() );  // needed for instanced offsets 
    548 
    549     LOG(trace) << "OGeo::makeTriangulatedGeometry "
    550               << " mmIndex " << mm->getIndex()
    551               << " numFaces (PrimitiveCount) " << numFaces
    552               << " numSolids " << numSolids
    553               << " numITransforms " << numITransforms
    554               ;
    555 
    556 
    557     GBuffer* id = NULL ;
    558     if(numITransforms > 0)
    559     {
    560         id = mm->getFaceRepeatedInstancedIdentityBuffer();
    561         assert(id);
    562         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer"
    563                   << " friid items " << id->getNumItems()
    564                   << " numITransforms*numFaces " << numITransforms*numFaces
    565                   ;
    566 
    567         assert( id->getNumItems() == numITransforms*numFaces );
    568    }
    569    else
    570    {
    571         id = mm->getFaceRepeatedIdentityBuffer();
    572         assert(id);
    573         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
    574                   << " frid items " << id->getNumItems()
    575                   << " numFaces " << numFaces
    576                   ;
    577         assert( id->getNumItems() == numFaces );
    578    }
    579 
    580    optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer");
    581    geometry["identityBuffer"]->setBuffer(identityBuffer);



FaceRepeatedIdentityBuffer
-----------------------------

::

    delta:opticks blyth$ opticks-find FaceRepeatedIdentityBuffer
    ./ggeo/GMesh.cc:GBuffer* GMesh::makeFaceRepeatedIdentityBuffer()
    ./ggeo/GMesh.cc:        LOG(warning) << "GMesh::makeFaceRepeatedIdentityBuffer only relevant to non-instanced meshes " ;
    ./ggeo/GMesh.cc:    LOG(info) << "GMesh::makeFaceRepeatedIdentityBuffer"
    ./ggeo/GMesh.cc:GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    ./ggeo/GMesh.cc:         m_facerepeated_identity_buffer = makeFaceRepeatedIdentityBuffer() ;  
    ./ggeo/tests/GGeoTest.cc:        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        id = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
    ./ggeo/GMesh.hh:      GBuffer* getFaceRepeatedIdentityBuffer(); 
    ./ggeo/GMesh.hh:      GBuffer* makeFaceRepeatedIdentityBuffer();




Face repeated from the solid level m_identity::

    1884 GBuffer* GMesh::makeFaceRepeatedIdentityBuffer()
    1885 {
    ....
    1902     guint4* nodeinfo = getNodeInfo();
    ....
    1916     // duplicate nodeinfo for each solid out to each face
    1917     unsigned int offset(0);
    1918     guint4* rid = new guint4[numFaces] ;
    1919     for(unsigned int s=0 ; s < numSolids ; s++)
    1920     {  
    1921         guint4 sid = m_identity[s]  ;
    1922         unsigned int nf = (nodeinfo + s)->x ;
    1923         for(unsigned int f=0 ; f < nf ; ++f) rid[offset+f] = sid ;
    1924         offset += nf ;
    1925     } 
    1926    
    1927     unsigned int size = sizeof(guint4) ;
    1928     GBuffer* buffer = new GBuffer( size*numFaces, (void*)rid, size, 4 );
    1929     return buffer ;
    1930 }


    1935 GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    1936 {
    1937     if(m_facerepeated_identity_buffer == NULL)
    1938     {
    1939          m_facerepeated_identity_buffer = makeFaceRepeatedIdentityBuffer() ;
    1940     }
    1941     return m_facerepeated_identity_buffer ;
    1942 }
    1943 

    delta:optixrap blyth$ opticks-find getFaceRepeatedIdentityBuffer 
    ./ggeo/GMesh.cc:GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    ./ggeo/tests/GGeoTest.cc:        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        id = mm->getFaceRepeatedIdentityBuffer();
    ./ggeo/GMesh.hh:      GBuffer* getFaceRepeatedIdentityBuffer(); 


Solid level identity are merged into m_identity within GMergedMesh methods such as::

    398 void GMergedMesh::mergeSolid( GSolid* solid, bool selected )
    399 {
    400     GMesh* mesh = solid->getMesh();
    401     unsigned int nvert = mesh->getNumVertices();
    402     unsigned int nface = mesh->getNumFaces();
    403     guint4 _identity = solid->getIdentity();
    ...
    411 
    412    if(m_verbosity > 1)
    413    {
    414 
    415         const char* pvn = solid->getPVName() ;
    416         const char* lvn = solid->getLVName() ;
    417 
    418         LOG(info) << "GMergedMesh::mergeSolid"
    419                   << " m_cur_solid " << m_cur_solid
    420                   << " idx " << solid->getIndex()
    421                   << " id " << _identity.description()
    422                   << " pv " << ( pvn ? pvn : "-" )
    423                   << " lv " << ( lvn ? lvn : "-" )
    424                   << " bb " << bb.description()
    425                   ;
    426         transform->Summary("GMergedMesh::mergeSolid transform");
    427    }
    428 
    429 
    430     unsigned int boundary = solid->getBoundary();
    431     NSensor* sensor = solid->getSensor();
    432 
    433     unsigned int nodeIndex = solid->getIndex();
    434     unsigned int meshIndex = mesh->getIndex();
    435     unsigned int sensorIndex = NSensor::RefIndex(sensor) ;
    436     assert(_identity.x == nodeIndex);
    437     assert(_identity.y == meshIndex);
    438     assert(_identity.z == boundary);
    439     //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
    440    


::

     920 void GMesh::setIdentity(guint4* identity)
     921 {
     922     m_identity = identity ;
     923     assert(m_num_solids > 0);
     924     unsigned int size = sizeof(guint4);
     925     assert(size == sizeof(unsigned int)*4 );
     926     m_identity_buffer = new GBuffer( size*m_num_solids, (void*)m_identity, size, 4 );
     927 }

::

    delta:ggeo blyth$ opticks-find setIdentity
    ./ggeo/GMesh.cc:    setIdentity(new guint4[numSolids]);
    ./ggeo/GMesh.cc:    if(strcmp(name, identity_) == 0)        setIdentityBuffer(buffer) ; 
    ./ggeo/GMesh.cc:void GMesh::setIdentity(guint4* identity)  
    ./ggeo/GMesh.cc:void GMesh::setIdentityBuffer(GBuffer* buffer) 
    ./ggeo/GTreeCheck.cc:     // cf GMesh::setIdentity
    ./ggeo/GMesh.hh:      void setIdentityBuffer(GBuffer* buffer);
    ./ggeo/GMesh.hh:      void setIdentity(guint4* identity);
    delta:opticks blyth$ 







From cache, see only node level identity, vaguely recall that face repeating is done dynamically and not persisted::

    In [1]: import numpy as np

    In [2]: i = np.load("identity.npy")

    In [3]: i
    Out[3]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ..., 
           [12227,   243,   122,     0],
           [12228,   244,   122,     0],
           [12229,   245,   122,     0]], dtype=uint32)

    In [4]: i.shape
    Out[4]: (12230, 4)

    In [5]: ii = np.load("iidentity.npy")

    In [6]: ii.shape
    Out[6]: (12230, 4)

    In [7]: ii
    Out[7]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ..., 
           [12227,   243,   122,     0],
           [12228,   244,   122,     0],
           [12229,   245,   122,     0]], dtype=uint32)



