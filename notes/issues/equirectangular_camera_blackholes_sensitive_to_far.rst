equirectangular_camera_blackholes_sensitive_to_far
=======================================================


issue 1
--------

::

    geocache-
    geocache-360 

    B,B : to see the 20-inch PMT instance bounding boxes and then the PMTs
    O   : swich to COMPOSITE raytrace + projective
    D,D : switch to ORTHOGRAPHIC and then EQUIRECTANGULAR

    A great big black hole is apparent with a curved edge and PMTs around the side

    F : then mousing up and down, shows a small movement of the edge of the black hole
    N : then mousing up decreases the size of the black hole quickly        
        but when increase further it acts to cut off PMTs from the other edge 

    geocache-360 () 
    { 
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        local cvd=1;
        UseOptiX --cvd $cvd;
        $dbg OKTest --cvd $cvd --envkey --xanalytic --target 62594 --eye 0,0,0 --tracer --look 1,0,0 --up 0,0,1 --enabledmergedmesh 2 $*
    }





First try to fix
-------------------

1. add Composition::hasNoRasterizedRender that skips the compositing when using EQUIRECTANGULAR_CAMERA
   but it made no difference 

   EQUIRECTANGULAR is a raytrace only camera style for now, because it is just too difficult with 
   a rasterized approach, 

   I was thinking that mismatched camera style depth information from the rasterized frame was having 
   an effect in the compositing 
   

Second try worked, simply use fixed depth for the raytrace
-------------------------------------------------------------

Simply use fixed depth for the raytrace with EQUIRECTANGULAR_CAMERA as there is no rasterized equivalent to composite with anyhow

::

   36     prd.result = make_float4(intensity, intensity, intensity, parallel == 2u ? 0.5f : zHit_clip ); // hijack .w for the depth, see notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst  




Review the depth calculation and how compositing works
-------------------------------------------------------

* compositing relies on having a projection matrix  



optixrap/cu/material1_radiance.cu::

     01 #include "switches.h"
      2 
      3 #include <optix.h>
      4 #include <optix_math.h>
      5 #include "PerRayData_radiance.h"
      6 
      7 //geometric_normal is set by the closest hit intersection program 
      8 rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
      9 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
     10 
     11 rtDeclareVariable(float3, contrast_color, , );
     12 
     13 rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
     14 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     15 rtDeclareVariable(float, t,            rtIntersectionDistance, );
     16 
     17 rtDeclareVariable(unsigned int,  touch_mode, , );
     18 rtDeclareVariable(float4,        ZProj, , );
     19 rtDeclareVariable(float3,        front, , );
     20 rtDeclareVariable(unsigned int,  parallel, , );
     21 
     22 
     23 RT_PROGRAM void closest_hit_radiance()
     24 {
     25     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     26     const float cos_theta = dot(n,ray.direction);
     27 
     28     float intensity = 0.5f*(1.0f-cos_theta) ;  // lambertian 
     29 
     30     float zHit_eye = -t*dot(front, ray.direction) ;   // intersect z coordinate (eye frame), always -ve 
     31     float zHit_ndc = parallel == 0 ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
     32     float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles
     33 
     34     //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );
     35 
     36     prd.result = make_float4(intensity, intensity, intensity, zHit_clip ); // hijack alpha for the depth 
     37 
     38 #ifdef BOOLEAN_DEBUG
     39      switch(instanceIdentity.x)
     40      {
     41         case 1: prd.result.x = 1.f ; break ;
     42         case 2: prd.result.y = 1.f ; break ;
     43         case 3: prd.result.z = 1.f ; break ;
     44     }
     45 #endif
     46 
     47     prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code
     48 }
     49 


::

    [blyth@localhost issues]$ opticks-f ZProj
    ./ana/geocache.bash:    2019-04-15 10:47:32.820 INFO  [150689] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-17316.9) front 0.5824,0.8097,-0.0719
    ./optickscore/Camera.cc:void Camera::fillZProjection(glm::vec4& zProj)
    ./optickscore/Camera.hh:     void fillZProjection(glm::vec4& zProj);
    ./optickscore/Composition.cc:    m_camera->fillZProjection(zproj);
    ./optickscore/Composition.cc:void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj )
    ./optickscore/Composition.cc:    m_camera->fillZProjection(ZProj); // 3rd row of projection matrix
    ./optickscore/Composition.cc:    glm::vec4 ZProj ;
    ./optickscore/Composition.cc:    getEyeUVW(eye,U,V,W,ZProj);
    ./optickscore/tests/CameraTest.cc:    c->fillZProjection(zpers);
    ./optickscore/tests/CameraTest.cc:    c->fillZProjection(zpara);
    ./optickscore/tests/CompositionTest.cc:   cam->fillZProjection(zproj);
    ./optickscore/Composition.hh:      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj);
    ./optixrap/cu/material1_radiance.cu:rtDeclareVariable(float4,        ZProj, , );
    ./optixrap/cu/material1_radiance.cu:    float zHit_ndc = parallel == 0 ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
    ./optixrap/cu/material1_radiance.cu:    //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );
    ./optixrap/OTracer.cc:    glm::vec4 ZProj ;
    ./optixrap/OTracer.cc:    m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
    ./optixrap/OTracer.cc:    m_context[ "ZProj"  ]->setFloat( make_float4( ZProj.x, ZProj.y, ZProj.z, ZProj.w ) );
    ./optixrap/OTracer.cc:                   << " ZProj.zw (" <<  ZProj.z << "," <<  ZProj.w << ")"
    [blyth@localhost opticks]$ 

::

    094 void OTracer::trace_()
     95 {
     96     LOG(debug) << "OTracer::trace_ " << m_trace_count ;
     97 
     98     double t0 = BTimeStamp::RealTime();  // THERE IS A HIGHER LEVEL WAY TO DO THIS
     99 
    100     glm::vec3 eye ;
    101     glm::vec3 U ;
    102     glm::vec3 V ;
    103     glm::vec3 W ;
    104     glm::vec4 ZProj ;
    105 
    106     m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
    107 
    108     unsigned parallel = m_composition->getParallel();  // 0:PERSP, 1:ORTHO, 2:EQUIRECT
    109     unsigned pixeltime_style = m_composition->getPixelTimeStyle() ;
    110     float    pixeltime_scale = m_composition->getPixelTimeScale() ;
    111     float      scene_epsilon = m_composition->getNear();
    112 
    113     const glm::vec3 front = glm::normalize(W);
    114 
    115     m_context[ "parallel"]->setUint( parallel );
    116     m_context[ "pixeltime_style"]->setUint( pixeltime_style );
    117     m_context[ "pixeltime_scale"]->setFloat( pixeltime_scale );
    118     m_context[ "scene_epsilon"]->setFloat(scene_epsilon);
    119     m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    120     m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    121     m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    122     m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );
    123     m_context[ "front"  ]->setFloat( make_float3( front.x, front.y, front.z ) );
    124     m_context[ "ZProj"  ]->setFloat( make_float4( ZProj.x, ZProj.y, ZProj.z, ZProj.w ) );
    125 
    126     Buffer buffer = m_context["output_buffer"]->getBuffer();
    127     RTsize buffer_width, buffer_height;
    128     buffer->getSize( buffer_width, buffer_height );



::

    2019 void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj )
    2020 {
    2021     update();
    2022 
    2023 
    2024     bool parallel = m_camera->getParallel();
    ////      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  POSSIBLE CAUSE

    2025     float scale = m_camera->getScale();
    2026     float length   = parallel ? scale : m_gazelength ;
    ////      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  POSSIBLE CAUSE :   BUT WHAT SHOULD LENGTH BE FOR EQUIRECTANGULAR_CAMERA
    ///       actually lots of the camera parameters make no sense in equirectangular   
    2027 
    2028    /*
    2029     float near  = m_camera->getNear();  
    2030     float basis = m_camera->getBasis() ; 
    2031     LOG(info) 
    2032          << " parallel " << parallel 
    2033          << " scale " << scale 
    2034          << " basis " << basis 
    2035          << " near " << near 
    2036          << " m_gazelength " << m_gazelength 
    2037          << " length " << length
    2038          ;
    2039     */
    2040 
    2041 
    2042     float tanYfov = m_camera->getTanYfov();  // reciprocal of camera zoom
    2043     float aspect = m_camera->getAspect();
    2044 
    2045     m_camera->fillZProjection(ZProj); // 3rd row of projection matrix
    2046 
    2047     //float v_half_height = m_gazelength * tanYfov ;  
    2048     float v_half_height = length * tanYfov ;
    2049     float u_half_width  = v_half_height * aspect ;
    2050     float w_depth       = m_gazelength ;
    2051 
    2052     //  Eye frame axes and origin 
    2053     //  transformed into world frame
    2054 
    2055     glm::vec4 right( 1., 0., 0., 0.);
    2056     glm::vec4   top( 0., 1., 0., 0.);
    2057     glm::vec4  gaze( 0., 0.,-1., 0.);
    2058 
    2059     glm::vec4 origin(0., 0., 0., 1.);
    2060 
    2061     // and scaled to focal plane dimensions 
    2062 
    2063     U = glm::vec3( m_eye2world * right ) * u_half_width ;
    2064     V = glm::vec3( m_eye2world * top   ) * v_half_height ;
    2065     W = glm::vec3( m_eye2world * gaze  ) * w_depth  ;
    2066 
    2067     eye = glm::vec3( m_eye2world * origin );
    2068 
    2069 }


::

    541 void Camera::fillZProjection(glm::vec4& zProj)
    542 {
    543     glm::mat4 proj = getProjection() ;
    544     zProj.x = proj[0][2] ;
    545     zProj.y = proj[1][2] ;
    546     zProj.z = proj[2][2] ;
    547     zProj.w = proj[3][2] ;
    548 }

