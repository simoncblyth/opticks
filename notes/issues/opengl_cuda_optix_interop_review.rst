opengl_cuda_optix_interop_review
====================================




CUDA OpenGL interop
----------------------

::

    epsilon:opticks blyth$ opticks-f cuda_gl_interop
    ./cudarap/CResource_.cu:#include <cuda_gl_interop.h>
    ./externals/cuda.bash:   find $(cuda-samples-dir) -type f -exec grep -${2:-l} ${1:-cuda_gl_interop.h} {} \;  
    ./thrustrap/thrap.bash:    #include <cuda_gl_interop.h>
    ./examples/ThrustOpenGLInterop/thrust_opengl_interop.cu:#include <cuda_gl_interop.h>
    epsilon:opticks blyth$ 




CUDA samples
--------------

* https://docs.nvidia.com/cuda/cuda-samples/index.html#simple-opengl

/usr/local/cuda/samples/2_Graphics/simpleGL



Refs
-----


* https://on-demand.gputechconf.com/gtc/2012/presentations/S0267A-GTC2012-Mixing-Graphics-Compute.pdf


Simple OpenGL-CUDA interop sample

Use mapping hint with cudaGraphicsResourceSetMapFlags() cudaGraphicsMapFlagsReadOnly/cudaGraphicsMapFlagsWriteDiscard:

::


    GLuint imagePBO;
    cudaGraphicsResource_t cudaResourceBuf;
    //OpenGL buffer creation
    glGenBuffers(1, &imagePBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, imagePBO); 
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,0);
    //Registration with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaResourceBuf, imagePBO, cudaGraphicsRegisterFlagsNone);


    GLuint imageTex;
    cudaGraphicsResource_t cudaResourceTex;
    //OpenGL texture creation
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    //set texture parameters here
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA8UI_EXT, width, height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    //Registration with CUDA
    cudaGraphicsGLRegisterImage (&cudaResourceTex, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);



    unsigned char *memPtr;
    cudaGraphicsResourceSetMapFlags(cudaResourceBuf, cudaGraphicsMapFlagsWriteDiscard)   ## mapping hint 
    while (!done) 
    {
         cudaGraphicsMapResources(1, &cudaResourceBuf, cudaStream); 
         cudaGraphicsResourceGetMappedPointer((void **)&memPtr, &size, cudaResourceBuf); 
         doWorkInCUDA(memPtr, cudaStream);//asynchronous 
         cudaGraphicsUnmapResources(1, &cudaResourceBuf, cudaStream); 
         doWorkInGL(imagePBO); //asynchronous
    }
 


OptiX 7 OpenGL interop
------------------------

* https://forums.developer.nvidia.com/t/opengl-in-optix-7/83142

dhart Oct 2019::

OptiX 6.5 and earlier needs explicit interop since the handles you get from
OptiX are not native device pointers. Since OptiX 7 uses raw device pointers
and explicit CUDA streams, like you pointed out, OptiX doesn’t need any interop
functionality beyond the interop you can find with CUDA. This goes for OpenGL
as well as DirectX and Vulkan too. So the way to think about it is with OptiX
6.5 and earlier you need OpenGL-OptiX interop, and with OptiX 7+ you need
OpenGL-CUDA interop. There is still an explicit API for interop, but the API is
CUDA functions rather than OptiX functions.

All three of those APIs have some limitations in their interop with CUDA since
they all deal in opaque buffer objects like OptiX used to. So you just need to
draw from the knowledge base of OpenGL-CUDA interop in order to use OpenGL with
OptiX 7. This set of slides is a bit dated, but I think still relevant to
OpenGL-CUDA interop https://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf


* https://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf

What Every CUDA Programmer Should
Know About OpenGL
The Fairmont San Jose | 4:00 PM Thursday, October 1 2009 | Joe Stam


EGL : NVIDIA off screen OpenGL with no X server
--------------------------------------------------

See env- egl- egl--


OpenGL CUDA Interop
---------------------

* https://forums.developer.nvidia.com/t/modern-gl-interop/44615

* https://on-demand.gputechconf.com/gtc/2012/presentations/S0267A-GTC2012-Mixing-Graphics-Compute.pdf

OpenGL PBO that are registered with CUDA

* eg the ray trace buffer from OptiX can be passed as texture into OpenGL  


Compositing
--------------

::

    1041 void Scene::render()
    1042 {
    1043     //LOG(info) << desc() ; 
    1044     m_composition->update();  // Oct 2018, moved prior to raytrace render
    1045 
    1046     bool raytraced = m_composition->isRaytracedRender() ;
    1047     bool composite = m_composition->isCompositeRender() ;
    1048     bool norasterized = m_composition->hasNoRasterizedRender() ;
    1049 
    1050     if(raytraced || composite)
    1051     {
    1052         if(m_raytrace_renderer)
    1053             m_raytrace_renderer->render() ;
    1054 
    1055         if(raytraced) return ;
    1056         if(composite && norasterized) return ;  // didnt fix notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst
    1057     }
    1061 
    1062     const glm::vec4& lodcut = m_composition->getLODCut();
    1063     const glm::mat4& world2eye = m_composition->getWorld2Eye();
    1064     const glm::mat4& world2clip = m_composition->getWorld2Clip();
    1065     m_context->update( world2clip, world2eye , lodcut );
    1066 
    ....
    1077     preRenderCompute();
    1078     renderGeometry();
    1079     renderEvent();
    1080 
    1081     m_render_count++ ;
    1082 }

    0988 void Scene::renderGeometry()
     989 {
     990     if(m_skipgeo_style == NOSKIPGEO )
     991     {
     992         if(*m_global_mode_ptr && m_global_renderer)       m_global_renderer->render();
     993         if(*m_globalvec_mode_ptr && m_globalvec_renderer) m_globalvec_renderer->render();
     994         // hmm this could be doing both global and globalvec ? Or does it need to be so ?
     995 
     996 
     997         for(unsigned int i=0; i<m_num_instance_renderer; i++)
     998         {
     999             if(m_instance_mode[i] && m_instance_renderer[i]) m_instance_renderer[i]->render();
    1000             if(m_bbox_mode[i] && m_bbox_renderer[i])         m_bbox_renderer[i]->render();
    1001         }
    1002     }
    1003 
    1004     if(m_axis_mode && m_axis_renderer)     m_axis_renderer->render();
    1005 }
    1006 
    1007 
    1008 void Scene::renderEvent()
    1009 {
    1010     if(m_skipevt_style == NOSKIPEVT )
    1011     {
    1012         if(m_genstep_mode && m_genstep_renderer)  m_genstep_renderer->render();
    1013         if(m_nopstep_mode && m_nopstep_renderer)  m_nopstep_renderer->render();
    1014         if(m_photon_mode  && m_photon_renderer)   m_photon_renderer->render();
    1015         if(m_source_mode  && m_source_renderer)   m_source_renderer->render();
    1016         if(m_record_mode)
    1017         {
    1018             Rdr* rdr = getRecordRenderer();
    1019             if(rdr)
    1020                 rdr->render();
    1021         }
    1022     }
    1023 }



zHit_clip depth info that allows compositing ray trace geometry with OpenGL event objects like photons
--------------------------------------------------------------------------------------------------------

* :google:`compositing ray trace with OpenGL`
* http://blog.wachowicz.eu/?p=21&cpage=3

3D (depth) composition of CUDA ray traced images with OpenGL rasterized images using CUDA Driver API


::

     35 
     36 rtDeclareVariable(unsigned int,  touch_mode, , );
     37 rtDeclareVariable(float4,        ZProj, , );     // Composition::getEyeUVW, fed in by OTracer::trace_
     38 rtDeclareVariable(float3,        front, , );     // normalized look direction, fed in by OTracer::trace_
     39 rtDeclareVariable(unsigned,      cameratype, , );  // camera type
     40 
     41 /**
     42 material1_radiance.cu:closest_hit_radiance
     43 -------------------------------------------
     44 
     45 Simple labertian shading used for ray trace images.
     46 
     47 *prd.result.w* provides the z-depth which is used to allow 
     48 compositing of raytrace images and rasterized images 
     49 
     50 **/
     51 
     52 
     53 RT_PROGRAM void closest_hit_radiance()
     54 {
     55     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     56     const float cos_theta = dot(n,ray.direction);
     57 
     58     float intensity = 0.5f*(1.0f-cos_theta) ;  // lambertian 
     59 
     60     float zHit_eye = -t*dot(front, ray.direction) ;   // intersect z coordinate (eye frame), always -ve 
     61     float zHit_ndc = cameratype == 0u ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
     62     float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles
     63 
     64     //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );
     65 
     66     prd.result = make_float4(intensity, intensity, intensity, cameratype == 2u ? 0.1f : zHit_clip );
     67     // hijack .w for the depth, see notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst  
     68 
     69 #ifdef BOOLEAN_DEBUG
     70      switch(instanceIdentity.x)
     71      {
     72         case 1: prd.result.x = 1.f ; break ;
     73         case 2: prd.result.y = 1.f ; break ;
     74         case 3: prd.result.z = 1.f ; break ;
     75     }
     76 #endif
     77 
     78     prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code
     79 }



optixrap/OTracer.cc pre 7 ray trace views of geometry
----------------------------------------------------------



