rcs-source(){   echo ${BASH_SOURCE} ; }
rcs-edir(){ echo $(dirname $(rcs-source)) ; }
rcs-ecd(){  cd $(rcs-edir); }

#rcs-dir(){  echo $HOME/rtx_compute_samples_6 ; }
rcs-dir(){  echo $HOME/rtx_compute_samples_7 ; }

rcs-cd(){   cd $(rcs-dir); }
rcs-vi(){   vi $(rcs-source) ; }
rcs-env(){  echo -n ; }
rcs-usage(){ cat << EOU

"Vishal Mehta (Compute DevTech)" <vishalm@nvidia.com>  mail of Jan 12, 2021

Regarding our discussion on step-by-step porting for Optix 6 code to Optix 7. I
don't have a formal presentation, but I have some code samples.  The repo here
has samples in both optix-7 (master branch) as well as legacy-optix-6 branch. 

Step 0: (SAXPY just raygen shader, setup pipeline for optix)
Optix 6: https://github.com/NVIDIA/rtx_compute_samples/tree/legacy-optix-6/optixSaxpy
Optix 7: https://github.com/NVIDIA/rtx_compute_samples/tree/master/optixSaxpy

Step 1: (Particle collision sample has all shaders)
Optix 6: https://github.com/NVIDIA/rtx_compute_samples/tree/legacy-optix-6/optixParticleCollision
Optix 7: https://github.com/NVIDIA/rtx_compute_samples/tree/master/optixParticleCollision

Hope this helps. Do not hesitate to reach out to us if you have questions.

Thanks
Vishal

::

    epsilon:rtx_compute_samples blyth$ git branch -a
    * master
      remotes/origin/HEAD -> origin/master
      remotes/origin/legacy-optix-6
      remotes/origin/master
    epsilon:rtx_compute_samples blyth$ 


rtx_compute_samples_7
    RTXDataHolder does some containment 


Suspect Optix7 more difficult to have large payload::

     69   // Extract Payload as unsigned int
     70   unsigned int ray_id_payload = pld.ray_id;
     71   unsigned int tpath_payload = __float_as_uint(pld.tpath);
     72 
     73   optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
     74              visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
     75              ray_id_payload, tpath_payload);
     76 

optix_7_device.h restricted to 8*32bit slots of payload::

    209 static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
    210                                                    float3                 rayOrigin,
    211                                                    float3                 rayDirection,
    212                                                    float                  tmin,
    213                                                    float                  tmax,
    214                                                    float                  rayTime,
    215                                                    OptixVisibilityMask    visibilityMask,
    216                                                    unsigned int           rayFlags,
    217                                                    unsigned int           SBToffset,
    218                                                    unsigned int           SBTstride,
    219                                                    unsigned int           missSBTIndex,
    220                                                    unsigned int&          p0,
    221                                                    unsigned int&          p1,
    222                                                    unsigned int&          p2,
    223                                                    unsigned int&          p3,
    224                                                    unsigned int&          p4,
    225                                                    unsigned int&          p5,
    226                                                    unsigned int&          p6,
    227                                                    unsigned int&          p7 );


optix_7_device.h::

    324 /// Returns the traversable handle for the Geometry Acceleration Structure (GAS) containing
    325 /// the current hit. May be called from IS, AH and CH.
    326 static __forceinline__ __device__ OptixTraversableHandle optixGetGASTraversableHandle();

    What use is that ? 

    388 static __forceinline__ __device__ unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle handle );

    For this ? This looks good : with OptiX6 the geometry structure used was to enable labelling the instances, 
    it looks like can have a simpler geometry structure and use this.


    527 /// Returns the OptixInstance::instanceId of the instance within the top level acceleration structure associated with the current intersection.
    528 ///
    529 /// When building an acceleration structure using OptixBuildInputInstanceArray each OptixInstance has a user supplied instanceId.
    530 /// OptixInstance objects reference another acceleration structure.  During traversal the acceleration structures are visited top down.
    531 /// In the Intersection and AH programs the OptixInstance::instanceId corresponding to the most recently visited OptixInstance is returned when calling optixGetInstanceId().
    532 /// In CH optixGetInstanceId() returns the OptixInstance::instanceId when the hit was recorded with optixReportIntersection.
    533 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns ~0u
    534 static __forceinline__ __device__ unsigned int optixGetInstanceId();

    535 
    536 /// Returns the zero-based index of the instance within its instance acceleration structure associated with the current intersection.
    537 ///
    538 /// In the Intersection and AH programs the index corresponding to the most recently visited OptixInstance is returned when calling optixGetInstanceIndex().
    539 /// In CH optixGetInstanceIndex() returns the index when the hit was recorded with optixReportIntersection.
    540 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns 0
    541 static __forceinline__ __device__ unsigned int optixGetInstanceIndex();


internal/optix_7_device_impl_exception.h::

    handy debug dump functions


optix7-pdf


p15

Multiple build inputs can be passed as an array to optixAccelBuild to combine
different meshes into a single acceleration structure. All build inputs for a
single build must agree on the build input type.

p19 

Primitives inside a build input are indexed starting from zero. This primitive
index is accessible inside the IS, AH and CH program.


Each instance description references: A 24-bit user-supplied ID


Q: Is it advantageous for all the GAS handles referenced from an IAS
   to be the same GAS ? Which would typically need a handful of separate IAS.

   Or instead could have one IAS that references a handful of different GAS 


::

    instance.visibilityMask = 255;   
    // 8 bits that gets OR with rayMask for fast customization of what is visible 
    // hmm so it would be good to have a maximum of 8 groups  
   


::

    513 /// For a given OptixBuildInputTriangleArray the number of primitives is defined as
    514 /// (OptixBuildInputTriangleArray::indexBuffer == nullptr) ? OptixBuildInputTriangleArray::numVertices/3 :
    515 ///                                                          OptixBuildInputTriangleArray::numIndices/3;
    516 ///
    517 /// For a given OptixBuildInputCustomPrimitiveArray the number of primitives is defined as
    518 /// numAabbs.  The primitive index returns is the index into the corresponding build array
    519 /// plus the primitiveIndexOffset.
    520 ///
    521 /// In Intersection and AH this corresponds to the currently intersected primitive.
    522 /// In CH this corresponds to the primitive index of the closest intersected primitive.
    523 /// In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT corresponds to the active primitive index. Returns zero for all other exceptions.
    524 static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex();
    525 


::

     348 typedef struct OptixBuildInputCustomPrimitiveArray
     349 {
     350     /// Points to host array of device pointers to AABBs (type OptixAabb), one per motion step.
     351     /// Host array size must match number of motion keys as set in OptixMotionOptions (or an array of size 1
     352     /// if OptixMotionOptions::numKeys is set to 1).
     353     /// Each device pointer must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.
     354     const CUdeviceptr* aabbBuffers;
     355 
     356     /// Number of primitives in each buffer (i.e., per motion step) in
     357     /// #OptixBuildInputCustomPrimitiveArray::aabbBuffers.
     358     unsigned int numPrimitives;
     359 
     360     /// Stride between AABBs (per motion key). If set to zero, the aabbs are assumed to be tightly
     361     /// packed and the stride is assumed to be sizeof( OptixAabb ).
     362     unsigned int strideInBytes;
     363 
     364     /// Array of flags, to specify flags per sbt record,
     365     /// combinations of OptixGeometryFlags describing the
     366     /// primitive behavior, size must match numSbtRecords
     367     const unsigned int* flags;
     368 
     369     /// Number of sbt records available to the sbt index offset override.
     370     unsigned int numSbtRecords;
     371 
     372     /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
     373     /// Every entry must be in range [0,numSbtRecords-1].
     374     /// Size needs to be the number of primitives.
     375     CUdeviceptr sbtIndexOffsetBuffer;
     376 
     377     /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
     378     unsigned int sbtIndexOffsetSizeInBytes;
     379 
     380     /// Stride between the index offsets. If set to zero, the offsets are assumed to be tightly
     381     /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
     382     unsigned int sbtIndexOffsetStrideInBytes;
     383 
     384     /// Primitive index bias, applied in optixGetPrimitiveIndex().
     385     /// Sum of primitiveIndexOffset and number of primitive must not overflow 32bits.
     386     unsigned int primitiveIndexOffset;
     387 } OptixBuildInputCustomPrimitiveArray;
     388 



p31

Symbols in OptixModule objects may be unresolved and contain extern references
to variables and __device__ functions. During pipeline creation, these symbols
can be resolved using the symbols defined in the pipeline modules. Duplicate
symbols will trigger an error.

p32 

all GPU instructions are allowed including math, texture, atomic operations,
control flow, and memory loads/stores
...
If needed, atomic operations may be used to share data between launch indices,
as long as an ordering between launch indices is not required. Memory fences
are not supported.


p37 

Q:what is a continuation stack ? Presumably the longest call chain between programs
 
p41 

The shader binding table (SBT) is an array of SBT records that hold information
about the location of programs and their parameters. The SBT resides in device
memory and is managed by the application.

An SBT record consists of a header and a data block. The header content is
opaque to the application. It holds information accessed by traversal execution
to identify and invoke programs. The data block is not used by NVIDIA OptiX and
holds arbitrary program-specific application information that is accessible in
the program. 

Q: how big can the data block be ?





Hmm with OptiX6 use the boundary to get offsets into the texture 


p53

Payloads are limited in size and are encoded in a maximum of 8 32-bit integer
values, which are held in registers when possible. These values may also encode
pointers to stack-based variables or application-managed global memory, to hold
additional state.

p55

The hitKind of the given intersection is communicated to the associated AH and
CH program and allows the AH and CH programs to customize how the attributes
should be interpreted. The lowest 7 bits of the hitKind are interpreted; values
[128, 255] are reserved for internal use.

Q: hmm perhaps use hitKind to pass the volume_index within a compound GAS ? 

   * too many volumes in global, but would be ok for the repeats
   
No more than eight values can be used for attributes. Unlike the ray payload
that can contain pointers to local memory, attributes should not contain
pointers to local memory. This memory may not be available in the CH or IS
shaders when the attributes are consumed.

p56 

The primitive index of the current intersection point can be queried using
optixGetPrimitiveIndex. The primitive index is local to its build input. 

Q: for a custom GAS with say 10 aabb this would give 0->9 ?


p58

The transform list contains all transforms on the path through the scene graph
from the root traversable passed to optixTrace to the current primitive.
Function optixGetTransformListSize returns the number of entries in the
transform list and optixGetTransformListHandle returns the traversable handle
of the transform entries.


p59 (example accessing the transform list) 


p60

The value returned by optixGetTransformListSize can be specialized with the
OptixPipelineCompileOptions::traversableGraphFlags compile option by selecting
which subset of traversables need to be supported. For example if only one
level of instancing is necessary and no motion blur transforms need to be
supported set traversableGraphFlags to
OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING.

p63

Two types of callable programs exist in NVIDIA OptiX 7: direct callables (DC)
and continuation callables (CC). Direct callables are called immediately,
similar to a regular CUDA function call, while continuation callables are
executed by the scheduler. Thus, a continuation callable can feature more
overhead when executed.




EOU
}
rcs-get(){
   local dir=$(dirname $(rcs-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://github.com/NVIDIA/rtx_compute_samples

   [ ! -d rtx_compute_samples_7 ] && git clone $url rtx_compute_samples_7
   [ ! -d rtx_compute_samples_6 ] && git clone --branch legacy-optix-6 $url rtx_compute_samples_6

}
