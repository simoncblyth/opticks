oxrap-hunt-for-f64-in-ptx
=============================

context :doc:`rtxmode-performance-jumps-by-factor-3-or-4-after-flipping-with-torus-switch-off`



Compare Release and Debug PTX on S
--------------------------------------

::

    [simon@localhost ~]$ opticks-ptx
                       BASH_SOURCE : /home/simon/opticks/opticks.bash 
                          FUNCNAME : opticks-ptx 
                               ptx : /data/simon/local/opticks_Debug/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx 
                  num_printf_lines : 95 
                     num_f64_lines : 518 


    [simon@localhost ~]$ opticks-ptx
                       BASH_SOURCE : /home/simon/opticks/opticks.bash 
                          FUNCNAME : opticks-ptx 
                               ptx : /data/simon/local/opticks_Release/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx 
                  num_printf_lines : 4 
                     num_f64_lines : 8 
    [simon@localhost ~]$ 

After some help from ~/opticks/preprocessor.sh to examine the flattened 
sources and some study of the PTX::

    [simon@localhost ~]$ opticks-ptx
                       BASH_SOURCE : /home/simon/opticks/opticks.bash 
                          FUNCNAME : opticks-ptx 
                               ptx : /data/simon/local/opticks_Release/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx 
                  num_printf_lines : 0 
                     num_f64_lines : 0 
    [simon@localhost ~]$ 





The doubles maybe from::

     37 template<> inline double scurand<double>::uniform( curandStateXORWOW* rng )
     38 {
     39 #ifdef FLIP_RANDOM
     40     return 1. - curand_uniform_double(rng) ;
     41 #else
     42     return curand_uniform_double(rng) ;
     43 #endif
     44 }



::

      402 
      403     setp.lt.u32     %p14, %r593, 51;
      404     @%p14 bra   $L__BB0_16;
      405 
      406     st.local.u32    [%rd7], %r587;
      407     cvt.ftz.f64.f32     %fd1, %f14;
      408     st.local.f64    [%rd7+8], %fd1;
      409     cvt.ftz.f64.f32     %fd2, %f16;
      410     cvt.ftz.f64.f32     %fd3, %f15;
      411     st.local.v2.f64     [%rd7+16], {%fd3, %fd2};
      412     cvt.ftz.f64.f32     %fd4, %f13;
      413     st.local.f64    [%rd7+32], %fd4;
      414     add.s32     %r1638, %r31, 1;
      415     st.local.v2.u32     [%rd7+40], {%r1638, %r24};
      416     mov.u64     %rd55, $str;
      417     cvta.global.u64     %rd56, %rd55;
      418     { // callseq 0, 0
      419     .reg .b32 temp_param_reg;
      420     .param .b64 param0;



      402 
      403     setp.lt.u32     %p14, %r593, 51;
      404     @%p14 bra   $L__BB0_16;
      405 
      406     st.local.u32    [%rd7], %r587;




      412     cvt.ftz.f64.f32     %fd4, %f13;
      413     st.local.f64    [%rd7+32], %fd4;

      407     cvt.ftz.f64.f32     %fd1, %f14;
      408     st.local.f64    [%rd7+8], %fd1;

      410     cvt.ftz.f64.f32     %fd3, %f15;
      409     cvt.ftz.f64.f32     %fd2, %f16;
      411     st.local.v2.f64     [%rd7+16], {%fd3, %fd2};

      /// store local v2.f64 (double2) from 2 f32 converted into f64 

      415     st.local.v2.u32     [%rd7+40], {%r1638, %r24};

      /// store local v2.u32 (uint2) from ? 

      414     add.s32     %r1638, %r31, 1;
      416     mov.u64     %rd55, $str;
      417     cvta.global.u64     %rd56, %rd55;


      418     { // callseq 0, 0
      419     .reg .b32 temp_param_reg;
      420     .param .b64 param0;







Update
----------


::

    ~/o/bin/ptx.py $(opticks-prefix)/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx

    N[blyth@localhost ~]$ ~/o/bin/ptx.py $(opticks-prefix)/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    ptx.py /home/blyth/junotop/ExternalLibs/opticks/head/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
     518 : TOTAL .f64 lines in function regions of the PTX 
     518 :  line:0090 : .visible .entry __raygen__rg()  
       0 :  line:12612 : .visible .entry __miss__ms()  
       0 :  line:12647 : .visible .entry __closesthit__ch()  
       0 :  line:13312 : .visible .entry __intersection__is()  
    N[blyth@localhost ~]$ 

    N[blyth@localhost ~]$ grep f64 $(opticks-prefix)/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx | wc -l 
    637

    N[blyth@localhost ~]$ grep printf $(opticks-prefix)/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx | wc -l 
    95


TODO: ptx report to compare before PRODUCTION switch 


Avoidable Sources of .f64 in OptiX 6.0.0. PTX 
------------------------------------------------

1. rtPrintExceptionDetails
2. rtPrintf of floats    
    ## aha : that explains why i see it in bounds at lot, I have a habit of leaving rtPrintf in bounds progs
    as they only get run one... 


Legitimate source of f64 : WITH_LOGDOUBLE
--------------------------------------------

::

     57 
     58 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     59 {
     60     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     61     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     62 
     63 #ifdef WITH_ALIGN_DEV
     64     float u_boundary_burn = curand_uniform(&rng) ;
     65     float u_scattering = curand_uniform(&rng) ;
     66     float u_absorption = curand_uniform(&rng) ;
     67 
     68 #ifdef WITH_LOGDOUBLE
     69     //  these two "log(double())" brings about 100 lines of PTX with .f64
     70     //  see notes/issues/AB_SC_Position_Time_mismatch.rst      
     71     float scattering_distance = -s.material1.z*log(double(u_scattering)) ;   // .z:scattering_length
     72     float absorption_distance = -s.material1.y*log(double(u_absorption)) ;   // .y:absorption_length 
     73 #else
     74     float scattering_distance = -s.material1.z*logf(u_scattering) ;   // .z:scattering_length
     75     float absorption_distance = -s.material1.y*logf(u_absorption) ;   // .y:absorption_length 
     76 #endif
     77 
     78 #else
     79     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     80     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     81 #endif
     82 
     83 #ifdef WITH_ALIGN_DEV_DEBUG
     84     rtPrintf("propagate_to_boundary  u_OpBoundary:%.9g speed:%.9g \n", u_boundary_burn, speed );
     85     rtPrintf("propagate_to_boundary  u_OpRayleigh:%.9g   scattering_length(s.material1.z):%.9g scattering_distance:%.9g \n", u_scattering, s.material1.z, scattering_distance );
     86     rtPrintf("propagate_to_boundary  u_OpAbsorption:%.9g   absorption_length(s.material1.y):%.9g absorption_distance:%.9g \n", u_absorption, s.material1.y, absorption_distance );
     87 #endif
     88 





Hunting for .f64 in OptiX 6.0.0 samples
--------------------------------------------

precompiled
~~~~~~~~~~~~~~~~

::

    [blyth@localhost ptx]$ pwd
    /home/blyth/local/opticks/externals/OptiX/SDK-precompiled-samples/ptx

    [blyth@localhost ptx]$ grep \\.f64 *.ptx  | wc -l
    170


    [blyth@localhost ptx]$ grep -l \\.f64 *.ptx  
    optixBuffersOfBuffers_generated_pinhole_camera.cu.ptx
    optixCallablePrograms_generated_pinhole_camera.cu.ptx
    optixConsole_generated_pinhole_camera.cu.ptx
    optixDynamicGeometry_generated_pinhole_camera.cu.ptx
    optixInstancing_generated_pinhole_camera.cu.ptx
    optixMDLDisplacement_generated_pinhole_camera.cu.ptx
    optixMeshViewer_generated_pinhole_camera.cu.ptx
    optixMotionBlur_generated_pinhole_camera.cu.ptx
    optixPrimitiveIndexOffsets_generated_pinhole_camera.cu.ptx
    optixSelector_generated_pinhole_camera.cu.ptx
    optixSphere_generated_pinhole_camera.cu.ptx
    optixSpherePP_generated_pinhole_camera.cu.ptx

    optixMDLExpressions_generated_mdl_material.cu.ptx
    optixMDLSphere_generated_camera.cu.ptx

    [blyth@localhost ptx]$ grep -l \\.f64 *pinhole_camera.cu.ptx | wc -l
    12
    [blyth@localhost ptx]$ l *pinhole_camera.cu.ptx | wc -l
    12

    ## most of them are from exception entry point 

    [blyth@localhost ptx]$ ptx.py --exclude exceptionv | c++filt
    /home/blyth/local/opticks/externals/OptiX_600/SDK-precompiled-samples/ptx/optixMDLSphere_generated_camera.cu.ptx
      10 : TOTAL 
      10 : 0107 : .visible .entry pinhole_camera()(  
       0 : 0659 : .visible .entry exception()(  
    /home/blyth/local/opticks/externals/OptiX_600/SDK-precompiled-samples/ptx/optixMDLExpressions_generated_mdl_material.cu.ptx
       4 : TOTAL 
       4 : 0142 : .visible .entry closest_hit_radiance()(  
       0 : 0651 : .visible .entry any_hit_shadow()(  
       0 : 0675 : .visible .entry miss()(  



built : had to switch if NVRTC off to have at look at PTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Building the samples with optix-samples-- after switching NVRTC off to have at look at PTX

::

    [blyth@localhost ptx]$ ptx.py  --exclude exception
    /home/blyth/local/opticks/externals/OptiX_600/SDK-src.build/lib/ptx/optixMDLSphere_generated_camera.cu.ptx
      10 : TOTAL 
      10 : 0107 : .visible .entry _Z14pinhole_camerav(  
       0 : 0577 : .visible .entry _Z9exceptionv(  
    [blyth@localhost ptx]$ 
    [blyth@localhost ptx]$ 


    [blyth@localhost ptx]$ l *pinhole*
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixSpherePP_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixSelector_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixSphere_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixPrimitiveIndexOffsets_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixMotionBlur_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixInstancing_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixMeshViewer_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 48094 Jan 26 03:51 optixMDLDisplacement_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixConsole_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixDynamicGeometry_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 48094 Jan 26 03:51 optixCallablePrograms_generated_pinhole_camera.cu.ptx
    -rw-r--r--. 1 blyth blyth 47329 Jan 26 03:51 optixBuffersOfBuffers_generated_pinhole_camera.cu.ptx
    [blyth@localhost ptx]$ 




finding f64 in oxrap PTX
----------------------------

::

    [blyth@localhost PTX]$ t oxrap-f64   ## counting lines with ".f64" in regions of the PTX
    oxrap-f64 is a function
    oxrap-f64 () 
    { 
        ptx.py $(opticks-prefix)/installcache/PTX --exclude exception | c++filt
    }



before going thru the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@localhost PTX]$ oxrap-f64
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_sphere_test.cu.ptx
      20 : TOTAL 
      20 : 0078 : .visible .entry intersect_analytic_sphere_test()(  
       0 : 0420 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_bufferTest.cu.ptx
      27 : TOTAL 
       9 : 0083 : .visible .entry bufferTest()(  
       0 : 0159 : .visible .entry bufferTest_0()(  
       9 : 0211 : .visible .entry bufferTest_1()(  
       9 : 0281 : .visible .entry bufferTest_2()(  
       0 : 0351 : .visible .entry bufferTest_3()(  
       0 : 0418 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_cbrtTest.cu.ptx
     109 : TOTAL 
     109 : 0080 : .visible .entry cbrtTest()(  
       0 : 0492 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_compactionTest.cu.ptx
       7 : TOTAL 
       7 : 0084 : .visible .entry compactionTest()(  
       0 : 0174 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_texTest.cu.ptx
      13 : TOTAL 
      13 : 0084 : .visible .entry texTest()(  
       0 : 0175 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_test.cu.ptx
     707 : TOTAL 
     707 : 0094 : .visible .entry intersect_analytic_test()(  
       0 : 2569 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/UseOContextBufferPP_generated_bufferTest.cu.ptx
       9 : TOTAL 
       9 : 0081 : .visible .entry bufferTest()(  
       0 : 0157 : .visible .entry printTest0()(  
       0 : 0212 : .visible .entry printTest1()(  
       0 : 0267 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_convexpolyhedron_test.cu.ptx
      15 : TOTAL 
      15 : 0075 : .visible .entry intersect_analytic_convexpolyhedron_test()(  
       0 : 0696 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_torus_test.cu.ptx
     707 : TOTAL 
     707 : 0094 : .visible .entry intersect_analytic_torus_test()(  
       0 : 2569 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_cone_test.cu.ptx
      20 : TOTAL 
      20 : 0077 : .visible .entry intersect_analytic_cone_test()(  
       0 : 0655 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
     227 : TOTAL 
       0 : 0223 : .visible .entry nothing()(  
       0 : 0234 : .visible .entry dumpseed()(  
       0 : 0313 : .visible .entry trivial()(  
       3 : 0418 : .visible .entry zrngtest()(  
       0 : 0661 : .visible .entry tracetest()(  
     224 : 1495 : .visible .entry generate()(  
       0 : 5691 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx
     297 : TOTAL 
     292 : 0137 : .visible .entry bounds(int, float*)(  
       5 : 3109 : .visible .entry intersect(int)(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_Roots3And4Test.cu.ptx
     326 : TOTAL 
     326 : 0080 : .visible .entry Roots3And4Test()(  
       0 : 1151 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_visit_instance.cu.ptx
      15 : TOTAL 
       0 : 0057 : .visible .entry visit_instance()(  
      15 : 0083 : .visible .entry visit_instance_WORLD()(  
    /home/blyth/local/opticks/installcache/PTX/UseOptiXRapBufferPP_generated_bufferTest.cu.ptx
       9 : TOTAL 
       9 : 0081 : .visible .entry bufferTest()(  
       0 : 0157 : .visible .entry printTest0()(  
       0 : 0212 : .visible .entry printTest1()(  
       0 : 0267 : .visible .entry exception()(  
    /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_textureTest.cu.ptx
      21 : TOTAL 
      21 : 0073 : .visible .entry textureTest()(  
       0 : 0241 : .visible .entry exception()(  
    [blyth@localhost PTX]$ 



after are down to unavoidables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    blyth@localhost issues]$ oxrap-f64
    ptx.py /home/blyth/local/opticks/installcache/PTX --exclude exception
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_cbrtTest.cu.ptx
     109 : TOTAL 
     109 : 0080 : .visible .entry cbrtTest()(  
       0 : 0492 : .visible .entry exception()(  
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_torus_test.cu.ptx
     707 : TOTAL 
     707 : 0094 : .visible .entry intersect_analytic_torus_test()(  
       0 : 2569 : .visible .entry exception()(  
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_Roots3And4Test.cu.ptx
     326 : TOTAL 
     326 : 0080 : .visible .entry Roots3And4Test()(  
       0 : 1151 : .visible .entry exception()(  
    [blyth@localhost issues]$ 



develop ptx.py tool to list f64 by entry point in PTX files or dirs
------------------------------------------------------------------------

::

    cp /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx /tmp/1.ptx
    cd /tmp

    [blyth@localhost tmp]$ grep .visible 1.ptx | c++filt
    .visible .entry raygen()(
    .visible .entry closest_hit_radiance0()(
    .visible .entry miss()(
    .visible .entry printTest0()(
    .visible .entry printTest1()(
    .visible .entry exception()(


minimal understanding to be able to read PTX to some extent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    struct PerRayData_radiance
    {
      float3 result;           // 3*4 = 12
      float  importance;       // 1*4    4   
      int depth;               // 1*4    4      20 bytes 
    };


    RT_PROGRAM void miss()
    {
      prd_radiance.result = make_float3(1.f, 1.f, 1.f) ;
    }

    031 .global .align 4 .b8 prd_radiance[20];    // twenty bytes


    247     // .globl   _Z4missv
    248 .visible .entry _Z4missv(
    249 
    250 )
    251 {
    252     .reg .b32   %r<2>;          // delcare 2 registers %r0 Rr1 of 32 bits  
    253     .reg .b64   %rd<2>;         // declare 2 registers %rd0 %rd1 of 64 bits 
    254 
    255 
    256     mov.u64     %rd1, 1065353216;

    In [28]: np.float32(1).view(np.uint32)
    Out[28]: 1065353216


    257     st.global.u32   [prd_radiance+4], %rd1;     // 
    258     st.global.u32   [prd_radiance], %rd1;
    259     mov.u32     %r1, 1065353216;
    260     st.global.u32   [prd_radiance+8], %r1;
    261     ret;
    262 }



revisit : the hunt for f64
-------------------------------

* having rtPrintf in the code but without print enabled in runtime still adding f64 to PTX

::

    [blyth@localhost optickscore]$ OpticksSwitchesTest 
    2019-09-23 21:34:14.063 INFO  [66724] [main@30] WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 

    blyth@localhost optickscore]$ ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx | c++filt
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
     202 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0228 : .visible .entry nothing()(  
       0 :  line:0239 : .visible .entry dumpseed()(  
       0 :  line:0318 : .visible .entry trivial()(  
       0 :  line:0423 : .visible .entry zrngtest()(  
       0 :  line:0653 : .visible .entry tracetest()(  
     189 :  line:1487 : .visible .entry generate()(  
      13 :  line:5428 : .visible .entry exception()(  
    [blyth@localhost optickscore]$ 


Comment WITH_ALIGN_DEV_DEBUG and rebuild::

    [blyth@localhost cu]$ OpticksSwitchesTest
    2019-09-23 21:35:49.711 INFO  [78655] [main@30] WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 

    [blyth@localhost cu]$ ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx | c++filt
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
     116 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0212 : .visible .entry nothing()(  
       0 :  line:0223 : .visible .entry dumpseed()(  
       0 :  line:0302 : .visible .entry trivial()(  
       0 :  line:0407 : .visible .entry zrngtest()(  
       0 :  line:0637 : .visible .entry tracetest()(  
     103 :  line:1471 : .visible .entry generate()(  
      13 :  line:4710 : .visible .entry exception()(  
    [blyth@localhost cu]$ 


Comment WITH_LOGDOUBLE and rebuild::

    [blyth@localhost opticks]$ OpticksSwitchesTest
    2019-09-23 21:38:22.272 INFO  [91560] [main@30] WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_KLUDGE_FLAT_ZERO_NOPEEK 


    [blyth@localhost opticks]$ ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx | c++filt
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
      13 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0212 : .visible .entry nothing()(  
       0 :  line:0223 : .visible .entry dumpseed()(  
       0 :  line:0302 : .visible .entry trivial()(  
       0 :  line:0407 : .visible .entry zrngtest()(  
       0 :  line:0637 : .visible .entry tracetest()(  
       0 :  line:1471 : .visible .entry generate()(  
      13 :  line:4510 : .visible .entry exception()(  
    [blyth@localhost opticks]$ 



Add WITH_EXCEPTION switch::

    701 RT_PROGRAM void exception()
    702 {
    703     //const unsigned int code = rtGetExceptionCode();
    704 #ifdef WITH_EXCEPTION
    705     rtPrintExceptionDetails();
    706 #endif
    707     photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
    708 }
    709 


Gets down to zero f64::

    [blyth@localhost cudarap]$ ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx | c++filt
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
       0 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0192 : .visible .entry nothing()(  
       0 :  line:0203 : .visible .entry dumpseed()(  
       0 :  line:0282 : .visible .entry trivial()(  
       0 :  line:0387 : .visible .entry zrngtest()(  
       0 :  line:0617 : .visible .entry tracetest()(  
       0 :  line:1451 : .visible .entry generate()(  
       0 :  line:4490 : .visible .entry exception()(  
    [blyth@localhost cudarap]$ 

                
Put back WITH_LOGDOUBLE, gets to 103 lines with f64::

    [blyth@localhost opticks]$ OpticksSwitchesTest 
    2019-09-23 22:00:19.720 INFO  [159869] [main@30] WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 

    [blyth@localhost opticks]$  ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx | c++filt
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
     103 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0192 : .visible .entry nothing()(  
       0 :  line:0203 : .visible .entry dumpseed()(  
       0 :  line:0282 : .visible .entry trivial()(  
       0 :  line:0387 : .visible .entry zrngtest()(  
       0 :  line:0617 : .visible .entry tracetest()(  
     103 :  line:1451 : .visible .entry generate()(  
       0 :  line:4690 : .visible .entry exception()(  
    [blyth@localhost opticks]$ 



