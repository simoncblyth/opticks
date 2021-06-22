device_side_genstep_overhaul
=====================================

Currently all genstep struct get seen by compiler/runtime
----------------------------------------------------------------

oxrap/generate.cu::

    590 
    591     if(gencode == OpticksGenstep_G4Cerenkov_1042 )
    592     {
    593         CerenkovStep cs ;
    594         csload(cs, genstep_buffer, genstep_offset, genstep_id);
    595 #ifdef DEBUG
    596         if(dbg) csdebug(cs);
    597 #endif
    598         generate_cerenkov_photon(p, cs, rng );
    599         s.flag = CERENKOV ;
    600     }
    601     else if(gencode == OpticksGenstep_DsG4Scintillation_r3971 )
    602     {
    603         ScintillationStep ss ;
    604         ssload(ss, genstep_buffer, genstep_offset, genstep_id);
    605 #ifdef DEBUG
    606         if(dbg) ssdebug(ss);
    607 #endif
    608         generate_scintillation_photon(p, ss, rng );  // maybe split on gencode ?
    609         s.flag = SCINTILLATION ;
    610     }
    611     else if(gencode == OpticksGenstep_G4Scintillation_1042 )
    612     {
    613         Genstep_G4Scintillation_1042 ss ;
    614         ss.load( genstep_buffer, genstep_offset, genstep_id);
    615 #ifdef DEBUG
    616         if(dbg) ss.debug();
    617 #endif
    618         ss.generate_photon(p, rng );
    619         s.flag = SCINTILLATION ;
    620     }
    621     else if(gencode == OpticksGenstep_TORCH)
    622     {
    623         TorchStep ts ;
    624         tsload(ts, genstep_buffer, genstep_offset, genstep_id);
    625 #ifdef DEBUG
    626         if(dbg) tsdebug(ts);
    627 #endif
    628         generate_torch_photon(p, ts, rng );
    629         s.flag = TORCH ;
    630     }
    631     else if(gencode == OpticksGenstep_EMITSOURCE)
    632     {
    633         // source_buffer is input only, photon_buffer output only, 
    634         // photon_offset is same for both these buffers
    635 
    636         const float4* _source_buffer = &source_buffer[0] ;
    637 
    638         pload(p, _source_buffer, photon_offset );
    639 
    640         p.flags.u.x = 0u ;   // scrub any initial flags, eg when running from an input photon  
    641         p.flags.u.y = 0u ;
    642         p.flags.u.z = 0u ;
    643         p.flags.u.w = 0u ;
    644 



Whats wrong with this ?
--------------------------

1. do not like that multiple types that are not in use (different versions of S and C gensteps) need to be "seen" by compiler/runtime

   * approach does not scale to lots of types

2. using the scintillation genstep to handle reemission is going to complicate this because the genstep needs
   to hang around for the life of the photon not go out of scope after generation as in above example

3. much of genstep loading is in common between different types of genstep, currently the code to do that is duplicated

4. for each photon only **one** genstep is ever relevant

   * hmm not quite true: photons from other gensteps may undergo reemission ?
   * but scintillation generation/reemission doesnt depend much on the genstep it is mostly random




Alternative approaches
-------------------------

* generate struct/func with templated S and C struct types

  * that works with pure CUDA, see QPoly 
  * BUT: needs to work with an extern "C" __raygen_rg function so templating not viable

* one genstep "union" struct for all types which branches on gentype

  * at photon level there is always only **one** active genstep so having multiple structs for different types is nasty  
  * nice and simple
  * could use preprocessor pruning to pick between different S or C versions, avoiding bloat   
  
* code generation picking between the alternative versioned types 



CUDA template type
--------------------

* https://stackoverflow.com/questions/19864920/cuda-c-templating-of-kernel-parameter

* https://forums.developer.nvidia.com/t/how-to-run-templatized-global-function-cuda-templates/508/2

* https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/

* https://codereview.stackexchange.com/questions/193367/cuda-c-host-device-polymorphic-class-implementation

* https://stackoverflow.com/questions/20073452/alternative-way-to-template-struct-with-static-member-function-in-cuda



qudarap/QPoly.cu explores using templated __global__ functions and virtual methods with CUDA
------------------------------------------------------------------------------------------------

::

     02 #include <stdio.h>
      3 #include "qpoly.h"
      4 
      5 __global__ void _QPoly_demo()
      6 {
      7     RectangleV1 r1 ; r1.set_param(10.0, 10.0) ;
      8     RectangleV2 r2 ; r2.set_param(10.0, 10.0) ;
      9     TriangleV1 t1  ; t1.set_param(10.0, 10.0) ;
     10     TriangleV2 t2  ; t2.set_param(10.0, 10.0) ;
     11 
     12     printf(" r1.area %10.3f  r2.area %10.3f t1.area %10.3f t2.area %10.3f \n", r1.area(), r2.area(), t1.area(), t2.area() );
     13 }
     14 
     15 extern "C" void QPoly_demo(dim3 numBlocks, dim3 threadsPerBlock )
     16 {
     17     _QPoly_demo<<<numBlocks,threadsPerBlock>>>();
     18 }
     19 
     20 
     21 template <typename R, typename T>
     22  __global__ void _QPoly_tmpl_demo()
     23 {
     24     R rtmpl ;
     25     rtmpl.set_param(10.0, 10.0) ;
     26 
     27     T ttmpl ;
     28     ttmpl.set_param(10.0, 10.0) ;
     29 
     30     printf(" rtmpl.area %10.3f  ttmpl.area %10.3f \n", rtmpl.area(), ttmpl.area() );
     31 }
     32 
     33 extern "C" void QPoly_tmpl_demo(dim3 numBlocks, dim3 threadsPerBlock )
     34 {
     35     _QPoly_tmpl_demo<RectangleV1, TriangleV1><<<numBlocks,threadsPerBlock>>>();
     36     _QPoly_tmpl_demo<RectangleV1, TriangleV2><<<numBlocks,threadsPerBlock>>>();
     37     _QPoly_tmpl_demo<RectangleV2, TriangleV1><<<numBlocks,threadsPerBlock>>>();
     38     _QPoly_tmpl_demo<RectangleV2, TriangleV2><<<numBlocks,threadsPerBlock>>>();
     39 }




Hmm, templated __global__ looks fine with pure-cuda but need to work with OptiX 7 raygen functions, cx/OptiX7Test.cu::

    098 extern "C" __global__ void __raygen__rg()
     99 {
    100     const uint3 idx = optixGetLaunchIndex();
    101     const uint3 dim = optixGetLaunchDimensions();
    102 
     
The extern "C" linkage means that cannot use templates.

* HMM : DOES THAT MEAN NEED TO RESORT TO CODE GENERATION ? 


Code Generation
------------------


optixrap/CMakeLists.txt already does some generation that flips switches different ways to 
allow to pick the appropriate PTX at runtime::

     25 set(generate_enabled_combinations
     26 +ANGULAR_ENABLED,+WAY_ENABLED
     27 +ANGULAR_ENABLED,-WAY_ENABLED
     28 -ANGULAR_ENABLED,+WAY_ENABLED
     29 -ANGULAR_ENABLED,-WAY_ENABLED
     30 )
     31 
     32 foreach(flags ${generate_enabled_combinations})
     33     set(srcfile ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu)
     34     set(outfile ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags}.cu)
     35     set(script  ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py)
     36     message(STATUS "flags:${flags} outfile:${outfile}")
     37     add_custom_command(
     38         OUTPUT  ${outfile}
     39         COMMAND ${script} ${srcfile} --flags="${flags}" --out=${outfile}
     40         DEPENDS ${srcfile}
     41     )
     42     list(APPEND CU_SOURCES ${outfile})
     43 endforeach()
     44 


::

    epsilon:optixrap blyth$ touch cu/generate.cu
    epsilon:optixrap blyth$ om
    === om-make-one : optixrap        /Users/blyth/opticks/optixrap                                /usr/local/opticks/build/optixrap                            
    [  1%] Generating generate_-ANGULAR_ENABLED,-WAY_ENABLED.cu
    [  2%] Generating generate_+ANGULAR_ENABLED,+WAY_ENABLED.cu
    INFO:__main__:path:/Users/blyth/opticks/optixrap/cu/generate.cu
    INFO:__main__:path:/Users/blyth/opticks/optixrap/cu/generate.cu
    INFO:__main__:flags:+ANGULAR_ENABLED,+WAY_ENABLED
    INFO:__main__:flags:-ANGULAR_ENABLED,-WAY_ENABLED
    INFO:__main__:writing to /usr/local/opticks/build/optixrap/generate_-ANGULAR_ENABLED,-WAY_ENABLED.cu 
    INFO:__main__:writing to /usr/local/opticks/build/optixrap/generate_+ANGULAR_ENABLED,+WAY_ENABLED.cu 
    [  3%] Generating generate_-ANGULAR_ENABLED,+WAY_ENABLED.cu
    [  3%] Generating generate_+ANGULAR_ENABLED,-WAY_ENABLED.cu
    INFO:__main__:path:/Users/blyth/opticks/optixrap/cu/generate.cu
    INFO:__main__:path:/Users/blyth/opticks/optixrap/cu/generate.cu
    INFO:__main__:flags:+ANGULAR_ENABLED,-WAY_ENABLED
    INFO:__main__:flags:-ANGULAR_ENABLED,+WAY_ENABLED
    INFO:__main__:writing to /usr/local/opticks/build/optixrap/generate_+ANGULAR_ENABLED,-WAY_ENABLED.cu 
    INFO:__main__:writing to /usr/local/opticks/build/optixrap/generate_-ANGULAR_ENABLED,+WAY_ENABLED.cu 
    [  5%] Building NVCC ptx file OptiXRap_generated_generate_+ANGULAR_ENABLED,+WAY_ENABLED.cu.ptx
    [  5%] Building NVCC ptx file OptiXRap_generated_generate_-ANGULAR_ENABLED,-WAY_ENABLED.cu.ptx
    [  6%] Building NVCC ptx file OptiXRap_generated_generate_+ANGULAR_ENABLED,-WAY_ENABLED.cu.ptx
    [  6%] Building NVCC ptx file OptiXRap_generated_generate_-ANGULAR_ENABLED,+WAY_ENABLED.cu.ptx
    [ 41%] Built target OptiXRap
    [ 45%] Built target OSensorLibTest
    [ 45%] Built target OCtx3dTest
    [ 49%] Built target ORngTest
    [ 49%] Built target interpolationTest


cu/preprocessor.py acts on the flags, commenting the generated source, eg::

    722 //// #ifdef ANGULAR_ENABLED 
    723 ////     PerRayData_angular_propagate prd ; 
    724 //// #else 
    725     PerRayData_propagate prd ;
    726 //// #endif 
    727 











