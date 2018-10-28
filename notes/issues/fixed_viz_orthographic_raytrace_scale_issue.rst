fixed_viz_orthographic_raytrace_scale_issue
===============================================

1. While in raytrace render mode (O), switching to orthographic projection (D) does not update render

   * FIXED by making Camera::nextStyle set m_changed via setParallel 

2. While in orthographic projection (D), switching between raytrace and rasterized (O) 
   shows a discrepancy between the geometries : with the rasterized one matching the event.
   It is as if the raytrace projection (there is no MVP for raytracing) is missing something. 
   The geometry is bigger in the raytrace, making the event look too small for the geometry

   * changing camera scale with the GUI slider, whilst in orthograhic 
     modifies event and geometry together with rasterized, BUT only the 
     event is changed in raytrace

   * changing camera zoom with the GUI slider, whilst in orthograhic 
     modifies event and geometry together with rasterized, and also with 
     raytrace : BUT the event looks shrunk relative to the geometry

   * looks like camera scale is not being fed somewhere is needs to go 

   * FIXED by using scale in place of gazelength in parallel Composition::getEyeUVW


3. raytrace geometry doesnt feel gui changed zoom, unless rotate (V) 
4. in parallel raytrace geometry doesnt feel gui changed scale at all, even whilst rotating 

   * FIXED by using scale in place of gazelength in parallel Composition::getEyeUVW


5. still have updating problem, changes to camera param not causing an updated raytrace


orthographic raytrace scale issue
-------------------------------------

oxrap-/cu/pinhole_camera.cu::

     08 rtDeclareVariable(float3,        eye, , );
      9 rtDeclareVariable(float3,        U, , );
     10 rtDeclareVariable(float3,        V, , );
     11 rtDeclareVariable(float3,        W, , );
     12 rtDeclareVariable(float3,        front, , );
     .. 
     16 rtDeclareVariable(unsigned int,  parallel, , );
     .. 
     26 rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
     27 rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
     ..
     44 RT_PROGRAM void pinhole_camera()
     45 {
     46 
     47   PerRayData_radiance prd;
     48   prd.flag = 0u ;
     49   prd.result = bad_color ;
     50 
     51   float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;
     52 
     53   optix::Ray ray = parallel == 0 ?
     54                        optix::make_Ray( eye                 , normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
     55                      :
     56                        optix::make_Ray( eye + d.x*U + d.y*V , normalize(W)                , radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
     57                      ;
     58 


oxrap/OTracer.cc::

    086 void OTracer::trace_()
     87 {
     88     LOG(debug) << "OTracer::trace_ " << m_trace_count ;
     89 
     90     double t0 = BTimer::RealTime();
     91 
     92     glm::vec3 eye ;
     93     glm::vec3 U ;
     94     glm::vec3 V ;
     95     glm::vec3 W ;
     96     glm::vec4 ZProj ;
     97 
     98     m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
     99 
    100     bool parallel = m_composition->getParallel();
    101     float scene_epsilon = m_composition->getNear();
    102 
    103     const glm::vec3 front = glm::normalize(W);
    104 
    105     m_context[ "parallel"]->setUint( parallel ? 1u : 0u);
    106     m_context[ "scene_epsilon"]->setFloat(scene_epsilon);
    107     m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    108     m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    109     m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    110     m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );
    111     m_context[ "front"  ]->setFloat( make_float3( front.x, front.y, front.z ) );
    112     m_context[ "ZProj"  ]->setFloat( make_float4( ZProj.x, ZProj.y, ZProj.z, ZProj.w ) );
    113 


okc/Composition.cc using scale instead of m_gazelength in parallel appears to work::

    1970 void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj )
    1971 {
    1972     update();
    1973 
    1974 
    1975     bool parallel = m_camera->getParallel();
    1976     float scale = m_camera->getScale(); 
    1977     float length   = parallel ? scale : m_gazelength ;
    1978    
    1979    /*
    1980     float near  = m_camera->getNear();  
    1981     float basis = m_camera->getBasis() ; 
    1982     LOG(info) 
    1983          << " parallel " << parallel 
    1984          << " scale " << scale 
    1985          << " basis " << basis 
    1986          << " near " << near 
    1987          << " m_gazelength " << m_gazelength 
    1988          << " length " << length
    1989          ;
    1990     */
    1991 
    1992 
    1993     float tanYfov = m_camera->getTanYfov();  // reciprocal of camera zoom
    1994     float aspect = m_camera->getAspect();
    1995 
    1996     m_camera->fillZProjection(ZProj); // 3rd row of projection matrix
    1997 
    1998     //float v_half_height = m_gazelength * tanYfov ;  
    1999     float v_half_height = length * tanYfov ;  
    2000     float u_half_width  = v_half_height * aspect ;
    2001     float w_depth       = m_gazelength ;
    2002 
    2003     //  Eye frame axes and origin 
    2004     //  transformed into world frame
    2005 
    2006     glm::vec4 right( 1., 0., 0., 0.);
    2007     glm::vec4   top( 0., 1., 0., 0.);
    2008     glm::vec4  gaze( 0., 0.,-1., 0.);
    2009 
    2010     glm::vec4 origin(0., 0., 0., 1.);
    2011 
    2012     // and scaled to focal plane dimensions 
    2013 
    2014     U = glm::vec3( m_eye2world * right ) * u_half_width ;  
    2015     V = glm::vec3( m_eye2world * top   ) * v_half_height ;
    2016     W = glm::vec3( m_eye2world * gaze  ) * w_depth  ;
    2017 
    2018     eye = glm::vec3( m_eye2world * origin );
    2019 
    2020 }


raytrace updating issue
-------------------------

Changes to camera param (zoom or scale) are not honoured in the raytrace, 
unless are rotating. 

* Being too aggressive with the laziness perhaps ?

Checking in oglrap/GUI.cc the ImGUI sliders change values via a pointer argument,
so it aint surprising that no update is noted.
As workaround add a "Camera Changed" button.



::

    158 void OKGLTracer::render()
    159 {   
    160     if(m_otracer && m_orenderer)
    161     {
    162         if(m_composition->hasChangedGeometry())
    163         {
    164             unsigned int scale = m_interactor->getOptiXResolutionScale() ;
    165             m_otracer->setResolutionScale(scale) ;
    166             m_otracer->trace_();
    167             m_oframe->push_PBO_to_Texture();
    168 
    169 /*
    170             if(m_trace_count == 0 )
    171             {
    172                 LOG(info) << "OKGLTracer::render snapping first raytrace frame " ; 
    173                 m_ocontext->snap();
    174             }
    175 */
    176             m_trace_count++ ;
    177         }
    178         else
    179         {
    180             // dont bother tracing when no change in geometry
    181         }
    182     }
    183 }  





