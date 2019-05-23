oxrap-hunt-for-f64-in-ptx
=============================

context :doc:`rtxmode-performance-jumps-by-factor-3-or-4-after-flipping-with-torus-switch-off`


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


