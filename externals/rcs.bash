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





EOU
}
rcs-get(){
   local dir=$(dirname $(rcs-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://github.com/NVIDIA/rtx_compute_samples

   [ ! -d rtx_compute_samples_7 ] && git clone $url rtx_compute_samples_7
   [ ! -d rtx_compute_samples_6 ] && git clone --branch legacy-optix-6 $url rtx_compute_samples_6

}
