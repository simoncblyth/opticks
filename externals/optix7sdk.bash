optix7sdk-vi(){ vi $BASH_SOURCE ; }
optix7sdk-env(){ echo -n ; }
optix7sdk-usage(){ cat << EOU 
optix7sdk review
=================


sutil::loadScene
    uses tinygltf to load model data and uploads 

    Q: Where is triangle vertices/indices ? buffers/meshes ?

    The WaterBottle.gltf has one buffer of 149412 bytes with 5 bufferViews
    of byte lengths : 20392 + 30588 + 40784 + 30588 + 27060 

    2549*2*4 = 20392   2549:VEC2:FLOAT             TEXCOORD_0
    2549*3*4 = 30588   2549:VEC3:FLOAT             NORMAL 
    2549*4*4 = 40784   2549:VEC4:FLOAT             TANGENT (apparently tangents often float4, why?)
    2549*3*4 = 30588   2549:VEC3:FLOAT             POSITION

    13530*2  = 27060   13530:SCALAR:UNSIGNED_SHORT INDICES?
    13530 % 3 == 0, 13530//3 = 4510 (number of triangles)

* UNSIGNED_SHORT INDICES IMPLY ONLY UP TO 0xffff//3 = 21845 ~21k triangles


sutil::Scene::addBuffer
    uploads data and appends CUdeviceptr into m_buffers



Grokking mesh handling
-------------------------

sutil/Scene.cpp::

     466             mesh->indices.push_back( bufferViewFromGLTF<uint32_t>( model, scene, gltf_primitive.indices ) );
     467             mesh->material_idx.push_back( gltf_primitive.material );
     468             std::cerr << "\t\tNum triangles: " << mesh->indices.back().count / 3 << std::endl;

* buffer_view.data is the CUdeviceptr 





Examples
----------

optixTriangle
   single tri only, no instancing
   optixTriangle.h : empty HitGroup data

   * HUH: separate HitGroupRecord for every submesh of every instance ??
     (that seems very duplicitous)

optixBoundValues
optixCallablePrograms
optixCompileWithTasks
optixCurves
optixCustomPrimitive
optixCutouts
optixDemandLoadSimple
optixDemandTexture
optixDenoiser
optixDynamicGeometry
optixDynamicMaterials
optixHair
optixHello
optixMeshViewer
optixModuleCreateAbort
optixMotionGeometry
optixMultiGPU
optixNVLink
optixOpticalFlow
optixPathTracer
optixRaycasting
optixSimpleMotionBlur
optixSphere
optixVolumeViewer
optixWhitted



SBT and instancing
--------------------

sutil/Scene Scene::createSBT
    hitgroup SBT records : num_instance*num_submesh*num_raytype

    * Q: why the duplication over instances ? 

CSGOptiX/SBT SBT::createHitgroup
    hitgroup SBT records : total number of enabled Prim in all enabled solids

    * crucially no duplication over instances


Upshot of some reading is that there is no right or wrong, the SBT 
is very flexible.  

Hookup is done via setting:: 

   OptixInstance.sbtOffset 
   optixTrace args 



SBT formula
-------------

::

    sbt-index = sbt-instance-offset
                + (sbt-geometry-acceleration-structure-index * sbt-stride-from-trace-call) 
                + sbt-offset-from-trace-call


Within CSGOptiX as only one ray type that becomes::

     sbt-index = sbt-instance-offset + sbt-geometry-acceleration-structure-index 


sbt-stride-from-trace-call, sbt-offset-from-trace-call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSGOptix/CSGOptiX7.cu::

    109     const unsigned SBToffset = 0u ;
    110     const unsigned SBTstride = 1u ;
    ...
    115     optixTrace(
    116             handle,
    117             ray_origin,
    118             ray_direction,
    119             tmin,
    120             tmax,
    121             rayTime,
    122             visibilityMask,
    123             rayFlags,
    124             SBToffset,
    125             SBTstride,
    126             missSBTIndex,
    127             p0, p1
    128             );



sbt-geometry-acceleration-structure-index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Manual*

Each geometry acceleration structure build input references at least one SBT
record. The first SBT geometry acceleration structure index for each geometry
acceleration structure build input is the prefix sum of the number of SBT
records. Therefore, the computed SBT geometry acceleration structure index is
dependent on the order of the build inputs.

*Code review*

CSGOptiX uses one GAS for each CSGSolid ("compound" of numPrim CSGPrim)
and that one GAS always has only one buildInput which references
numPrim SBT records which have "sbt-geometry-acceleration-structure-index" 
of (0,1,2,...,numPrim-1)  


CSGOptiX/GAS_Builder.cc::

    087 BI GAS_Builder::MakeCustomPrimitivesBI_11N(const CSGPrimSpec& ps)
     88 {
     89     assert( ps.device == true );
     90     assert( ps.stride_in_bytes % sizeof(float) == 0 );
    ...
    113     buildInputCPA.numSbtRecords = ps.num_prim ;                      // number of sbt records available to sbt index offset override. 
    114     buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;           // Device pointer to per-primitive local sbt index offset buffer, Every entry must be in range [0,numSb    tRecords-1]
    115     buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);     // Size of type of the sbt index offset. Needs to be 0,1,2 or 4    
    116     buildInputCPA.sbtIndexOffsetStrideInBytes = ps.stride_in_bytes ; // Stride between the index offsets. If set to zero, the offsets are assumed to be tightly packed.
    117     buildInputCPA.primitiveIndexOffset = ps.primitiveIndexOffset ;   // Primitive index bias, applied in optixGetPrimitiveIndex() see OptiX7Test.cu:__closesthit__ch


CSG/CSGFoundry.cc::

    2054     CSGPrim pr = {} ;
    2055 
    2056     pr.setNumNode(num_node) ;
    2057     pr.setNodeOffset(nodeOffset);
    2058     pr.setSbtIndexOffset(localPrimIdx) ;  // NB must be localPrimIdx, globalPrimIdx was a bug 
    2059     pr.setMeshIdx(-1) ;                // metadata, that may be set by caller 
    2060 

    096 CSGPrimSpec CSGPrim::MakeSpec( const CSGPrim* prim0,  unsigned primIdx, unsigned numPrim ) // static 
    097 {
    098     const CSGPrim* prim = prim0 + primIdx ;
     99 
    100     CSGPrimSpec ps ;
    101     ps.aabb = prim->AABB() ;
    102     ps.sbtIndexOffset = prim->sbtIndexOffsetPtr() ;
    103     ps.num_prim = numPrim ;
    104     ps.stride_in_bytes = sizeof(CSGPrim);
    105     ps.primitiveIndexOffset = primIdx ;
    106 


Each CPU side prim has pr.setSbtIndexOffset called on it, 
but where are they collected and uploaded ?  

No separate CPU side collection is done, instead the offsets 
are just accessed from consequtive uploaded CSGPrim 
with a sizeof(CSGPrim) stride. Configured in the buildInputCPA::

The offsets are::

   (0,1,2,..,numPrim-1) for a CSGSolid with numPrim CSGPrim



sbt-instance-offset
~~~~~~~~~~~~~~~~~~~~~

OptixInstance::sbtOffset
      (limited to 28 bits)


IAS_Builder::CollectInstances
   
    within loop over instances::

        bool found = gasIdx_sbtOffset.count(gasIdx) == 1 ; 
        unsigned sbtOffset = found ? gasIdx_sbtOffset.at(gasIdx) : sbt->getOffset(gasIdx, prim_idx ) ;

        // prim_idx:0u for sbt offset of the outer prim(aka layer) of the GAS

        instance.sbtOffset = sbtOffset ;   
        // map cache avoids calling this for every instance 


     Q: Hmm does that mean always get the outer prim of the GAS ? 
     A: No, the buildInput specifies numPrim SBT records 
        with offsets that point to them which is used in the 
        SBT dispatch formula. 

     20 struct HitGroupData   // effectively Prim 
     21 {
     22     int numNode ;
     23     int nodeOffset ;
     24 };

     * its easier somehow to think of the "instance.sbtOffset" as a pointer into the sbt record array, 
       that gets offset by the buildInput.sbtIndexOffsetBuffer offsets 


What to search for to assess SBT use of an app
-----------------------------------------------

IAS/OptiXInstance::

    sbtOffset       

    (does not need to depend on the instance, 
     can just point to start of the appropriate 
     GAS/buildInput SBT records, in CSGOptiX this 
     points to the rec for the out CSGPrim of the CSGSolid) 

GAS/buildInput::

    numSbtRecords   (>= 1, often 1) 

    [  (below are not needed when numSbtRecords=1) 
    sbtIndexOffsetBuffer 
    sbtIndexOffsetSizeInBytes
    sbtIndexOffsetStrideInBytes
    ]


quotes from the manual (shortened for readability):

1. Each GAS/buildInput references at least one SBT record. 
   The first SBT GAS index for each GAS buildInput is 
   the prefix sum of the number of SBT records.
   Therefore, the computed SBT GAS index is dependent on the 
   order of buildInputs.

   * SCB: using prefix sum implies index offsets must be 0-based 
     and local to their buildInput 

2. Because build input 1 references a single SBT record, a sbtIndexOffsetBuffer
   does not need to be specified for the geometry acceleration structure build.

   * SCB: no need for a buffer to just supply a zero offset 
   





SBT explanations
-------------------

* https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways


* https://forums.developer.nvidia.com/t/sbt-theoretical-quesions/179309/9

droettger::

    I use an IAS->GAS scene structure in my OptiX 7 examples and an SBT entry
    per instance because that allowed changing shaders (SBT hit record header) per
    instance without updating the IAS because nothing changed in the
    OptixInstances.  This is kind of wasteful when having few materials and many
    instances (e.g. millions).  I would recommend implementing that differently
    today to simplify the SBT handling.

    There is another elegant method to index into the SBT for the material and hold
    the per instance data separately.  The SBT hit records only need to contain the
    32 byte header information which selects the material programs defining the
    material behavior (bi-directional scattering distribution function, BSDF), but
    none of the input parameters.  Which material is used can be directly selected
    with the OptixInstance sbtOffset value.

    All other data required to define an OptixInstance’s geometry topology (vertex
    attributes and indices) and any other data like material parameters can be
    stored in separate device memory arrays and uniquely indexed by the user
    defined OptixInstance instanceId field which can be read inside the device code
    with optixGetInstanceId() which is available in IS, AH, and CH programs.  Note
    that there is also the optixGetInstanceIndex() which returns the zero based
    index inside an IAS. Means when the scene is using a single IAS as the root
    (implies OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING) that would
    be another unique per instance index which could be used to access custom data
    in device memory.




optixGetTriangleVertexData
-----------------------------

::

    .304 /// Return the object space triangle vertex positions of a given triangle in a Geometry
     305 /// Acceleration Structure (GAS) at a given motion time.
     306 /// To access vertex data, the GAS must be built using the flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
     307 ///
     308 /// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not contain motion, the
     309 /// time parameter is ignored.
     310 static __forceinline__ __device__ void optixGetTriangleVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float3 data[3]);

     598 /// Returns the Sbt GAS index of the primitive associated with the current intersection.
     599 ///
     600 /// In IS and AH this corresponds to the currently intersected primitive.
     601 /// In CH this corresponds to the Sbt GAS index of the closest intersected primitive.
     602 /// In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT corresponds to the sbt index within the hit GAS. Returns zero for all other exceptions.
     603 static __forceinline__ __device__ unsigned int optixGetSbtGASIndex();



    epsilon:SDK blyth$ find . -type f -exec grep -H optixGetTriangleVertexData {} \;
    ./optixMotionGeometry/optixMotionGeometry.cu:    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
    ./optixVolumeViewer/volume.cu:    optixGetTriangleVertexData(
    ./optixDynamicGeometry/optixDynamicGeometry.cu:    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
    epsilon:SDK blyth$ 


::

    150     const OptixTraversableHandle gas       = optixGetGASTraversableHandle();
    151     const unsigned int           gasSbtIdx = optixGetSbtGASIndex();
    152     const unsigned int           primIdx   = optixGetPrimitiveIndex();
    153     float3 vertices[3] = {};
    154     optixGetTriangleVertexData(
    155         gas,
    156         primIdx,
    157         gasSbtIdx,
    158         0,
    159         vertices );
    160 


optixGetInstanceTraversableFromIAS
-----------------------------------

::

     300 /// Return the traversable handle of a given instance in an Instance 
     301 /// Acceleration Structure (IAS)
     302 static __forceinline__ __device__ OptixTraversableHandle optixGetInstanceTraversableFromIAS( OptixTraversableHandle ias, unsigned int instIdx );


no examples::

    epsilon:SDK blyth$ find . -type f -exec grep -H optixGetInstanceTraversableFromIAS {} \;
    epsilon:SDK blyth$ 





Intersection Distance
-----------------------

* with triangles need to get this from builtin optix in CH, 
  unlike with custom geom where calculate it in IS and pass it to CH 


* https://forums.developer.nvidia.com/t/about-distance-acquisition-of-optix7-0/112677/2

For the intersection distance you query optixGetRayTmin() [typo,Tmax?] 
inside the closest hit program.  You’ll find its link to the API reference documentation via the
above OptiX 7 device-side functions link to OptIX 7 Programming Guide.  In
OptiX 6 and earlier that was the float variable with semantic
rtIntersectionDistance

::

     273 /// Returns the tmin passed into optixTrace.
     274 ///
     275 /// Only available in IS, AH, CH, MS
     276 static __forceinline__ __device__ float optixGetRayTmin();
     277 
     278 /// In IS and CH returns the current smallest reported hitT or the tmax passed into optixTrace if no hit has been reported
     279 /// In AH returns the hitT value as passed in to optixReportIntersection
     280 /// In MS returns the tmax passed into optixTrace
     281 /// Only available in IS, AH, CH, MS
     282 static __forceinline__ __device__ float optixGetRayTmax();







EOU
}

