rasterized_render_revive
==========================

Old workflow sources, only gives very high level help.
Prefer guidance from other places first.

oglrap/OpticksViz.cc
   steering

oglrap/Scene.cc

oglrap/Renderer.cc
   Renderer::upload 



look for OpenGL/glfw buffer handling hints in OptiX7 SDK
----------------------------------------------------------

::

    epsilon:SDK blyth$ find . -type f -exec grep -l cuda_gl_interop {} \;
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp
    ./optixCutouts/optixCutouts.cpp
    ./optixBoundValues/optixBoundValues.cpp
    ./optixCallablePrograms/optixCallablePrograms.cpp
    ./optixMotionGeometry/optixMotionGeometry.cpp
    ./optixMultiGPU/optixMultiGPU.cpp
    ./optixPathTracer/optixPathTracer.cpp
    ./optixWhitted/optixWhitted.cpp
    ./optixMeshViewer/optixMeshViewer.cpp
    ./optixModuleCreateAbort/optixModuleCreateAbort.cpp
    ./sutil/CUDAOutputBuffer.h
    ./optixVolumeViewer/optixVolumeViewer.cpp
    ./optixNVLink/optixNVLink.cpp
    ./optixDynamicGeometry/optixDynamicGeometry.cpp
    ./optixDynamicMaterials/optixDynamicMaterials.cpp
    epsilon:SDK blyth$ 


Association between VAO and multiple VBO ? 
--------------------------------------------

::

    epsilon:oglrap blyth$ grep BindVertexArray *.*
    InstLODCull.cc:    glBindVertexArray(m_forkVAO);
    InstLODCull.cc:    glBindVertexArray(m_workaroundVAO);
    InstLODCull.cc:    glBindVertexArray(vertexArray);
    Rdr.cc:    glBindVertexArray (m_vao);     
    Rdr.cc:        glBindVertexArray(VAO);
    Rdr.cc:    glBindVertexArray(m_vao);
    Rdr.cc:    glBindVertexArray(0);
    Renderer.cc:    glBindVertexArray (vao);     
    Renderer.cc:            glBindVertexArray ( m_use_lod ? m_vao[i] : m_vao_all );
    Renderer.cc:            glBindVertexArray ( m_vao_all );
    Renderer.cc:        glBindVertexArray ( m_vao[0] );
    Renderer.cc:    glBindVertexArray(0);
    epsilon:oglrap blyth$ 



* https://stackoverflow.com/questions/21652546/what-is-the-role-of-glbindvertexarrays-vs-glbindbuffer-and-what-is-their-relatio

It is also worth mentioning that the "current" binding for GL_ARRAY_BUFFER is not one of the states that VAOs track.

* HUH ? 

* https://community.khronos.org/t/vaos-and-gl-array-buffer-binding/66751/3

It is not misleading; it is true. It’s a common misconception about
GL_ARRAY_BUFFER and VAOs that trips a lot of people up. I put that note there
so that people would know that binding to GL_ARRAY_BUFFER does not affect VAO
state. VAO state is only affected by glEnable/DisableVertexAttribArray and
glVertexAttribPointer calls.

The fact that glVertexAttribPointer happens to get its buffer from what is
bound to GL_ARRAY_BUFFER does not mean that GL_ARRAY_BUFFER is VAO state.

The bind to GL_ARRAY_BUFFER itself will not affect VAO state; only calling
glVertexAttribPointer will affect VAO state.



Old shaders ~/o/oglrap/gl/index.rst
-------------------------------------

::

    colour = vec4( normal*0.5 + 0.5, 1.0 - Param.z ) ; 



Old viz top level
------------------

::

    epsilon:opticks blyth$ opticks-fl OpticksViz.hh 
    ./opticksgl/OKGLTracer.cc
    ./ok/OKMgr.cc
    ./ok/tests/VizTest.cc
    ./ok/OKPropagator.cc
    ./okg4/tests/OKX4Test.cc
    ./okg4/OKG4Mgr.cc
    ./npy/NConfigurable.hpp
    ./oglrap/CMakeLists.txt
    ./oglrap/AxisApp.cc
    ./oglrap/OpticksViz.cc
    epsilon:opticks blyth$ 



renderloop::

    573 void OpticksViz::renderLoop()
    574 {
    575     if(m_interactivity == 0 )
    576     {
    577         LOG(LEVEL) << "early exit due to InteractivityLevel 0  " ;
    578         return ;
    579     }
    580     LOG(LEVEL) << "enter runloop ";
    581 
    582     //m_frame->toggleFullscreen(true); causing blankscreen then segv
    583     m_frame->hintVisible(true);
    584     m_frame->show();
    585     LOG(LEVEL) << "after frame.show() ";
    586 
    587     unsigned count(0) ;
    588     bool exitloop(false);
    589 
    590     int renderlooplimit = m_ok->getRenderLoopLimit();
    591 
    592     while (!glfwWindowShouldClose(m_window) && !exitloop  )
    593     {
    594         m_frame->listen();
    595 
    596 #ifdef OPTICKS_NPYSERVER
    597         if(m_server) m_server->poll_one();
    598 #endif
    599 #ifdef WITH_BOOST_ASIO
    600         m_io_context.poll_one();
    601 #endif
    602 
    603         count = m_composition->tick();
    604 
    605         if(m_launcher)
    606         {
    607             m_launcher->launch(count);
    608         }
    609 
    610         if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)
    611         {



How was key interation hooked up previously ?
-----------------------------------------------

::

    void Interactor::key_pressed(unsigned int key)
    void Frame::handle_event(GLEQevent& event)


    void Frame::listen()
    {
        glfwPollEvents();

        GLEQevent event;
        while (gleqNextEvent(&event))
        {    
            if(m_dumpevent) dump_event(event);
            handle_event(event);
            gleqFreeEvent(&event);
        }    
    }


trackball
------------

* https://github.com/zhangbo-tj/trackball  GPL so dont bother looking 

* https://github.com/BrutPitt/virtualGizmo3D  BSD

* https://git.science.uu.nl/s.carter/animationviewer/-/tree/bvh2/3rd_party/imGuIZMO.quat-3.0


* https://github.com/ocornut/imgui


* https://www.codeproject.com/Articles/22040/Arcball-OpenGL-in-C

Arcball (also know as Rollerball) is probably the most intuitive method to view
three dimensional objects. The principle of the Arcball is based on creating a
sphere around the object and letting users to click a point on the sphere and
drag it to a different location. 


* http://rainwarrior.ca/dragon/arcball.html


How old Trackball used
------------------------

okc/Interactor.cc::

    277     else if( m_pan_mode )
    278     {
    279         m_trackball->pan_to(df*x,df*y,df*dx,df*dy);
    280     }
    281     else if( m_zoom_mode )  // bad name, actully z translate
    282     {
    283         m_trackball->zoom_to(df*x,df*y,df*dx,df*dy);
    284     }
    285     else if( m_rotate_mode )
    286     {
    287         m_trackball->drag_to(rf*x,rf*y,rf*dx,rf*dy);
    288     }


    468 void Interactor::key_pressed(unsigned int key)
    469 {

    540         case GLFW_KEY_R:
    541             m_rotate_mode = !m_rotate_mode ;
    542             break;
    543         case GLFW_KEY_S:
    544             m_scale_mode = !m_scale_mode ;
    545             break;

    558         case _pan_mode_key:
    559             pan_mode_key_pressed(modifiers);
    560             break;
    561         case GLFW_KEY_Y:
    562             y_key_pressed(modifiers);
    563             break;
    564         case GLFW_KEY_Z:
    565             z_key_pressed(modifiers);
    566             break;

    429 void Interactor::z_key_pressed(unsigned int modifiers)
    430 {
    431     if(modifiers & OpticksConst::e_option)
    432     {
    433         m_composition->setEyeGUI("Z-");
    434     }
    435     else
    436     {
    437         m_zoom_mode = !m_zoom_mode ;
    438     }
    439 }


::

    1818 glm::mat4& Composition::getWorld2Camera()  // just view, no trackballing
    1819 {
    1820      return m_world2camera ;
    1821 }
    1822 glm::mat4& Composition::getCamera2World()  // just view, no trackballing
    1823 {
    1824      return m_camera2world ;
    1825 }

    1990 void Composition::update()
    1991 {


    2032     m_view->getTransforms(m_model2world, m_world2camera, m_camera2world, m_gaze );   // model2world is input, the others are updated
    2033     //
    2034     // the eye2look look2eye pair allows trackball rot to be applied around the look 
    2035     // recall the eye frame, has eye at the origin and the object are looking 
    2036     // at (0,0,-m_gazelength) along -Z (m_gazelength is +ve) eye2look in the 
    2037     // translation to jump between frames, from eye/camera frame to a frame centered on the object of the look 
    2038     //
    2039     // camera and eye frames are the same
    2040     // 
    2041     m_gazelength = glm::length(m_gaze);
    2042     m_eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,m_gazelength));
    2043     m_look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-m_gazelength));
    2044 
    2045     m_trackball->getOrientationMatrices(m_trackballrot, m_itrackballrot);  // this is just rotation, no translation
    2046     m_trackball->getTranslationMatrices(m_trackballtra, m_itrackballtra);  // just translation  
    2047 
    2048     m_world2eye = m_trackballtra * m_look2eye * m_trackballrot * m_lookrotation * m_eye2look * m_world2camera ;           // ModelView
    2049 
    2050     m_eye2world = m_camera2world * m_look2eye * m_ilookrotation * m_itrackballrot * m_eye2look * m_itrackballtra ;          // InverseModelView

    ////  m_world2eye m_eye2world are confusing names as camera~eye in other usage
    ////  better m_ModelView m_InverseModelView reflecting the connection with OpenGL usage 

    2052     // NB the changing of frame as each matrix is multiplied 
    2053     // lookrotation coming after eye2look means that will rotate around the look point (not the eye point)
    2054     // also trackballrot operates around the look 
    2055     // then return back out to eye frame after look2eye where the trackballtra gets applied 
    2056     //
    2057     // NB the opposite order for the eye2world inverse
    2058     
    2059     m_projection = m_camera->getProjection();
    2060     
    2061     m_world2clip = m_projection * m_world2eye ;    //  ModelViewProjection
    2062     


DONE : SGLM::updateComposite equiv of above Composition::update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



DONE : Arcball in SGLM_Arcball.h
-------------------------------------

* https://oguz81.github.io/ArcballCamera/
* https://github.com/oguz81/ArcballCamera

* Properties of Quaternions


Arcball with quaternions
--------------------------

* https://research.cs.wisc.edu/graphics/Courses/559-f2001/Examples/Gl3D/arcball-gems.pdf

* ~/opticks_refs/ken_shoemake_arcball_rotation_control_gem.pdf

* https://github.com/Twinklebear/arcball-cpp


glm/glm/gtx/quaternion.inl::

    122     GLM_FUNC_QUALIFIER qua<T, Q> rotation(vec<3, T, Q> const& orig, vec<3, T, Q> const& dest)
    123     {
    124         T cosTheta = dot(orig, dest);
    125         vec<3, T, Q> rotationAxis;
    126 
    127         if(cosTheta >= static_cast<T>(1) - epsilon<T>()) {
    128             // orig and dest point in the same direction
    129             return quat_identity<T,Q>();
    130         }
    131 
    132         if(cosTheta < static_cast<T>(-1) + epsilon<T>())
    133         {
    134             // special case when vectors in opposite directions :
    135             // there is no "ideal" rotation axis
    136             // So guess one; any will do as long as it's perpendicular to start
    137             // This implementation favors a rotation around the Up axis (Y),
    138             // since it's often what you want to do.
    139             rotationAxis = cross(vec<3, T, Q>(0, 0, 1), orig);
    140             if(length2(rotationAxis) < epsilon<T>()) // bad luck, they were parallel, try again!
    141                 rotationAxis = cross(vec<3, T, Q>(1, 0, 0), orig);
    142 
    143             rotationAxis = normalize(rotationAxis);
    144             return angleAxis(pi<T>(), rotationAxis);
    145         }
    146 
    147         // Implementation from Stan Melax's Game Programming Gems 1 article
    148         rotationAxis = cross(orig, dest);
    149 
    150         T s = sqrt((T(1) + cosTheta) * static_cast<T>(2));
    151         T invs = static_cast<T>(1) / s;
    152 
    153         return qua<T, Q>(
    154             s * static_cast<T>(0.5f),
    155             rotationAxis.x * invs,
    156             rotationAxis.y * invs,
    157             rotationAxis.z * invs);
    158     }

   


thoughts on impl of controls
-----------------------------

::

    447 /**
    448 Interactor::key_pressed
    449 ------------------------
    450 
    451 Hmm it would be better if the interactor
    452 talked to a single umbrella class (living at lower level, not up here)
    453 for controlling all this.  
    454 Composition does that a bit but far from completely.
    455 
    456 The reason is that having a single controller that can be talked to 
    457 by various means avoids duplication. The means could include: 
    458 
    459 * via keyboard (here with GLFW)
    460 * via command strings 
    461 
    462 The problem is that too much state is residing too far up the heirarchy, 
    463 it should be living in generic fashion lower down.
    464 In MVC speak : lots of "M" is living in "V" 
    465 
    466 **/
    467 


cursor drag controls
----------------------

::

    253 void Interactor::cursor_drag(float x, float y, float dx, float dy, int ix, int iy )
    254 {
    255     m_changed = true ;
    256     //printf("Interactor::cursor_drag x,y  %0.5f,%0.5f dx,dy  %0.5f,%0.5f \n", x,y,dx,dy );
    257 
    258     float rf = 1.0 ;
    259     float df = m_dragfactor ;
    260 
    261     if( m_yfov_mode )
    262     {
    263         m_camera->zoom_to(df*x,df*y,df*dx,df*dy);
    264     }
    265     else if( m_near_mode )
    266     {
    267         m_camera->near_to(df*x,df*y,df*dx,df*dy);
    268     }
    269     else if( m_far_mode )
    270     {
    271         m_camera->far_to(df*x,df*y,df*dx,df*dy);
    272     }
    273     else if( m_scale_mode )
    274     {
    275         m_camera->scale_to(df*x,df*y,df*dx,df*dy);
    276     }
    277     else if( m_pan_mode )
    278     {
    279         m_trackball->pan_to(df*x,df*y,df*dx,df*dy);
    280     }
    281     else if( m_zoom_mode )  // bad name, actully z translate
    282     {
    283         m_trackball->zoom_to(df*x,df*y,df*dx,df*dy);
    284     }
    285     else if( m_rotate_mode )
    286     {
    287         m_trackball->drag_to(rf*x,rf*y,rf*dx,rf*dy);
    288     }
    289     else if( m_time_mode )
    290     {





Index buffer
-------------

* https://openglbook.com/chapter-3-index-buffer-objects-and-primitive-types.html






CUDA OpenGL interop examples
-------------------------------

/usr/local/cuda-10.1/samples/2_Graphics/simpleGL/simpleGL.cu::

    

review Opticks glfw code
--------------------------

::

    epsilon:opticks blyth$ opticks-fl glfw
    ./opticksgl/ORenderer.cc
    ./opticksgl/OFrame.cc
    ./opticksgl/OKGLTracer.cc


    ./bin/oks.bash
    ./bin/findpkg.py
    ./bin/pc.py
    ./bin/oc.bash
    ./ok/ok.bash
    ./externals/externals.bash
    ./externals/optixnote.bash
    ./externals/optix7.bash
    ./externals/glfw.bash
    ./externals/imgui.bash
    ./externals/gleq.bash
    ./numpyserver/numpyserver.bash
    ./sysrap/SGLFW.h
    ./sysrap/SGLFW_tests/SGLFW_tests.cc
    ./cmake/Modules/FindOpticksGLFW.cmake



    ./examples/UseOpticksGLFWSnap/UseOpticksGLFWSnap.cc
          Pops up an OpenGL window with a colorful rotating single triangle
          On pressing SPACE a ppm snapshot of the window is saved to file. 
          [ this uses ancient non-shader OpenGL] 

    ./examples/UseShader/UseShader.cc
          Pops up an OpenGL window with a colorful single triangle

    ./examples/UseGeometryShader/UseGeometryShader.cc
    ./examples/UseGeometryShader/build.sh
          rec_flying_point visualization of photon step point record array 

    ./examples/UseOpticksGLFWNoCMake/glfw_keyname.h
    ./examples/UseOpticksGLFWNoCMake/go.sh
    ./examples/UseOpticksGLFWNoCMake/UseOpticksGLFW.cc
           "oc" no longer maintained, so needs reworking 

    ./examples/ThrustOpenGLInterop/thrust_opengl_interop.cu
           SKIP : difficult to get thrust stuff and opengl stuff to compile together 

    ./examples/UseOpticksGLFWSPPM/UseOpticksGLFWSPPM.cc
           Ancient non-shader OpenGL checking use of SPPM to 
           save the screen buffer when press SPACE

    ./examples/UseOpticksGLEW/UseOpticksGLEW.cc
           Trivial GLEW CMake and GLEW version macro test


    ./examples/UseInstance/tests/OneTriangleTest.cc
    ./examples/UseInstance/tests/UseInstanceTest.cc
    ./examples/UseInstance/Frame.cc
    ./examples/UseInstance/Renderer.cc
    ./examples/UseInstance/Prog.cc
           Minimal example of OpenGL instancing, 
           default test pops up a window with 8 instanced triangles


    ./examples/UseOpticksGLFW/glfw_keyname.h
    ./examples/UseOpticksGLFW/UseOpticksGLFW.cc

          ~/o/examples/UseOpticksGLFW/go.sh
           demonstrate GLFW key callbacks with modifiers, ancient OpenGL 


    ./examples/UseOGLRap/UseOGLRap.cc
    ./examples/UseOGLRapMinimal/UseOGLRapMinimal.cc


    ./opticks.bash
    ./boostrap/BListenUDP.hh
    ./optixrap/OGeo.cc
    ./oglrap/Frame.hh
    ./oglrap/Interactor.cc
    ./oglrap/OGLRap_imgui.hh
    ./oglrap/oglrap.bash
    ./oglrap/GUI.cc
    ./oglrap/gleq.h
    ./oglrap/GUI.hh
    ./oglrap/tests/SceneCheck.cc
    ./oglrap/tests/TexCheck.cc
    ./oglrap/OpticksViz.cc
    ./oglrap/Texture.cc
    ./oglrap/Frame.cc
    ./oglrap/Pix.cc
    ./oglrap/RContext.cc
    ./oglrap/old_gleq.h
    ./oglrap/RBuf.hh
    epsilon:opticks blyth$ 


   
how was the interop between OptiX/CUDA and OpenGL organized ?
--------------------------------------------------------------

::

    534 /**
    535 OpticksViz::setExternalRenderer
    536 ---------------------------------
    537 
    538 Used from opticksgl/OKGLTracer.cc OKGLTracer::prepareTracer with::
    539 
    540     111     m_composition->setRaytraceEnabled(true);  // enables the "O" key to switch to ray trace
    541     114     m_viz->setExternalRenderer(this);
    542 
    543 The SRenderer pure virtual base protocol is just two methods *render* and *snap* 
    544 that only use standard types in the interface.
    545 
    546 The external renderer handles the optix ray trace render to buffer
    547 and thence to a texture that gets pushed to OpenGL.  
    548 
    549 **/
    550 
    551 void OpticksViz::setExternalRenderer(SRenderer* external_renderer)
    552 {
    553     m_external_renderer = external_renderer ;
    554 }
    555 
    556 void OpticksViz::render()
    557 {
    558     m_frame->viewport();
    559     m_frame->clear();
    560 
    561     if(m_composition->isRaytracedRender() || m_composition->isCompositeRender())
    562     {
    563         if(m_external_renderer) m_external_renderer->render();
    564     }
    565 
    566     m_scene->render();
    567 }
    568 



    087 /**
     88 OKGLTracer::prepareTracer
     89 ---------------------------
     90 
     91 Establishes connection between: 
     92 
     93 1. oxrap.OTracer m_otracer (OptiX) resident here
     94 2. oglrap.Scene OpenGL "raytrace" renderer (actually its just renders tex pushed to it)
     95 
     96 **/
     97 
     98 void OKGLTracer::prepareTracer()
     99 {
    100     if(m_hub->isCompute()) return ;
    101     if(!m_scene)
    102     {
    103         LOG(fatal) << "OKGLTracer::prepareTracer NULL scene ?"  ;
    104         return ;
    105     }
    106 
    107     Scene* scene = Scene::GetInstance();
    108     assert(scene);
    109 
    110     //scene->setRaytraceEnabled(true);  // enables the "O" key to switch to ray trace
    111     m_composition->setRaytraceEnabled(true);  // enables the "O" key to switch to ray trace
    112 
    113 
    114     m_viz->setExternalRenderer(this);
    115 
    116     unsigned int width  = m_composition->getPixelWidth();
    117     unsigned int height = m_composition->getPixelHeight();
    118 
    119     LOG(debug) << "OKGLTracer::prepareTracer plant external renderer into viz"
    120                << " width " << width
    121                << " height " << height
    122                 ;
    123 
    124     m_ocontext = m_ope->getOContext();
    125 
    126     optix::Context context = m_ocontext->getContext();
    127 
    128     m_oframe = new OFrame(context, width, height);
    129 
    130     context["output_buffer"]->set( m_oframe->getOutputBuffer() );
    131 
    132     m_interactor->setTouchable(m_oframe);
    133 
    134     Renderer* rtr = m_scene->getRaytraceRenderer();
    135 
    136     m_orenderer = new ORenderer(rtr, m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());
    137 
    138     m_otracer = new OTracer(m_ocontext, m_composition);
    139 
    140     //m_ocontext->dump("OKGLTracer::prepareTracer");
    141 }



Where is the tex pushing done ?

    178 double OKGLTracer::render()
    179 {
    180     double dt = -1. ;
    181     if(m_otracer && m_orenderer)
    182     {
    183         if(m_composition->hasChangedGeometry())
    184         {
    185             unsigned int scale = m_interactor->getOptiXResolutionScale() ;
    186             m_otracer->setResolutionScale(scale) ;
    187             dt = m_otracer->trace_();
    188             m_oframe->push_PBO_to_Texture();
    189 
    197             m_trace_count++ ;
    198         }
    199         else
    200         {
    201             // dont bother tracing when no change in geometry
    202         }
    203     }
    204     return dt ;
    205 }


Hmm the OFrame is very OptiX < 7::

    080 void OFrame::init(unsigned int width, unsigned int height)
     81 {
     82     m_width = width ;
     83     m_height = height ;
     84 
     85     // generates the m_pbo and m_depth identifiers and buffers
     86     m_output_buffer = createOutputBuffer_PBO(m_pbo, RT_FORMAT_UNSIGNED_BYTE4, width, height) ;
     87 


    123 optix::Buffer OFrame::createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height, boo    l /*depth*/)
    124 {
    125     Buffer buffer;
    126 
    127     glGenBuffers(1, &id);
    128     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id);
    129 
    130     size_t element_size ;
    131     m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    132 
    133     LOG(debug) << "OFrame::createOutputBuffer_PBO"
    134               <<  " element_size " << element_size
    135               ;
    136 
    137     assert(element_size == 4);
    138 
    139     unsigned int nbytes = element_size * width * height ;
    140 
    141     m_pbo_data = (unsigned char*)malloc(nbytes);
    142     memset(m_pbo_data, 0x88, nbytes);  // initialize PBO to grey 
    143 
    144     glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, m_pbo_data, GL_STREAM_DRAW);
    145     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    146 
    147     buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    148     buffer->setFormat(format);
    149     buffer->setSize( width, height );
    150 
    151     LOG(debug) << "OFrame::createOutputBuffer_PBO  element_size " << element_size << " size (" << width << "," << height <<     ") pbo id " << id ;
    152 
    153     return buffer;
    154 }


::

    164 void OFrame::push_Buffer_to_Texture(optix::Buffer& buffer, int buffer_id, int texture_id, bool depth)
    165 {
    166     RTsize buffer_width_rts, buffer_height_rts;
    167     buffer->getSize( buffer_width_rts, buffer_height_rts );
    168 
    169     int buffer_width  = static_cast<int>(buffer_width_rts);
    170     int buffer_height = static_cast<int>(buffer_height_rts);
    171 
    172     RTformat buffer_format = buffer->getFormat();
    ...
    190     assert(buffer_id > 0);
    191 
    192     glBindTexture(GL_TEXTURE_2D, texture_id );
    193
    194     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);
    195 
    196     RTsize elementSize = buffer->getElementSize();
    197     if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    198     else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    199     else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    200     else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    201 
    202 
    203     GLenum target = GL_TEXTURE_2D ;
    204     GLint level = 0 ;            // level-of-detail number. Level 0 is the base image level

    ...   format details
    262 
    263     glTexImage2D(target, level, internalFormat, buffer_width, buffer_height, border, format, type, data);
    264 
    265     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    266     //glBindTexture(GL_TEXTURE_2D, 0 );   get blank screen when do this here
    267 
    268 }


    174     //
    175     // glTexImage2D specifies mutable texture storage characteristics and provides the data
    176     //
    177     //    *internalFormat* 
    178     //         format with which OpenGL should store the texels in the texture
    179     //    *data*
    180     //         location of the initial texel data in host memory, 
    181     //         if a buffer is bound to the GL_PIXEL_UNPACK_BUFFER binding point, 
    182     //         texel data is read from that buffer object, and *data* is interpreted 
    183     //         as an offset into that buffer object from which to read the data. 
    184     //    *format* and *type*
    185     //         initial source texel data layout which OpenGL will convert 
    186     //         to the internalFormat
    187     // 
    188     // send pbo data to the texture
 




::

     77 void ORenderer::render()
     78 {
     79     LOG(debug) << "ORenderer::render " << m_render_count ;
     80 
     81     double t0 = BTimeStamp::RealTime();
     82 
     83     m_frame->push_PBO_to_Texture();
     84 
     85     double t1 = BTimeStamp::RealTime();
     86 
     87     if(m_renderer)
     88         m_renderer->render();
     89 
     90     double t2 = BTimeStamp::RealTime();
     91 
     92     m_render_count += 1 ;
     93     m_render_prep += t1 - t0 ;
     94     m_render_time += t2 - t1 ;
     95 
     96     glBindTexture(GL_TEXTURE_2D, 0 );
     97 
     98     if(m_render_count % 10 == 0) report("ORenderer::render");
     99 }




Note order:

1. OpenGL PBO created
2. OptiX "wrapper" output buffer created from the PBO
3. OptiX launch writes to output buffer








7.5 SDK examples with interop
----------------------------------

::

    epsilon:SDK blyth$ find . -type f -exec grep -l interop {} \;
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp
    ./optixCutouts/optixCutouts.cpp
    ./optixBoundValues/optixBoundValues.cpp
    ./optixCallablePrograms/optixCallablePrograms.cpp
    ./optixMotionGeometry/optixMotionGeometry.cpp
    ./optixMultiGPU/optixMultiGPU.cpp
    ./optixPathTracer/optixPathTracer.cpp
    ./optixWhitted/optixWhitted.cpp
    ./optixMeshViewer/optixMeshViewer.cpp
    ./optixModuleCreateAbort/optixModuleCreateAbort.cpp
    ./sutil/CUDAOutputBuffer.h
    ./support/tinygltf/json.hpp
    ./optixVolumeViewer/optixVolumeViewer.cpp
    ./optixNVLink/optixNVLink.cpp
    ./optixDynamicGeometry/optixDynamicGeometry.cpp
    ./optixDynamicMaterials/optixDynamicMaterials.cpp
    epsilon:SDK blyth$ 


     
optixMeshViewer
-----------------

::

    283 void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, const sutil::Scene& scene )
    284 {
    285 
    286     // Launch
    287     uchar4* result_buffer_data = output_buffer.map();
    288     params.frame_buffer        = result_buffer_data;
    289     CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
    290                 &params,
    291                 sizeof( whitted::LaunchParams ),
    292                 cudaMemcpyHostToDevice,
    293                 0 // stream
    294                 ) );
    295 
    296     OPTIX_CHECK( optixLaunch(
    297                 scene.pipeline(),
    298                 0,             // stream
    299                 reinterpret_cast<CUdeviceptr>( d_params ),
    300                 sizeof( whitted::LaunchParams ),
    301                 scene.sbt(),
    302                 width,  // launch width
    303                 height, // launch height
    304                 1       // launch depth
    305                 ) );
    306     output_buffer.unmap();
    307     CUDA_SYNC_CHECK();
    308 }


sutil/CUDAOutputBuffer.h looks reusable to handle the PBO that can be written by CUDA::

    230 template <typename PIXEL_FORMAT>
    231 PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map()
    232 {   
    233     if( m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P )
    234     {   
    235         // nothing needed
    236     }
    237     else if( m_type == CUDAOutputBufferType::GL_INTEROP  )
    238     {   
    239         makeCurrent();
    240         
    241         size_t buffer_size = 0u;
    242         CUDA_CHECK( cudaGraphicsMapResources ( 1, &m_cuda_gfx_resource, m_stream ) );
    243         CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
    244                     reinterpret_cast<void**>( &m_device_pixels ),
    245                     &buffer_size,
    246                     m_cuda_gfx_resource
    247                     ) );
    248     }
    249     else // m_type == CUDAOutputBufferType::ZERO_COPY
    250     {   
    251         // nothing needed
    252     }
    253     
    254     return m_device_pixels;
    255 }


::

    311 void displaySubframe(
    312         sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
    313         sutil::GLDisplay&                 gl_display,
    314         GLFWwindow*                       window )
    315 {
    316     // Display
    317     int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    318     int framebuf_res_y = 0;   //
    319     glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    320     gl_display.display(
    321             output_buffer.width(),
    322             output_buffer.height(),
    323             framebuf_res_x,
    324             framebuf_res_y,
    325             output_buffer.getPBO()
    326             );
    327 }




GLDisplay : Shader that reads from a texture
------------------------------------------------

* looks like can be adapted for reuse

sutil/GLDisplay.cpp::

    142 const std::string GLDisplay::s_vert_source = R"(
    143 #version 330 core
    144 
    145 layout(location = 0) in vec3 vertexPosition_modelspace;
    146 out vec2 UV;
    147 
    148 void main()
    149 {
    150     gl_Position =  vec4(vertexPosition_modelspace,1);
    151     UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
    152 }
    153 )";
    154 
    155 const std::string GLDisplay::s_frag_source = R"(
    156 #version 330 core
    157 
    158 in vec2 UV;
    159 out vec3 color;
    160 
    161 uniform sampler2D render_tex;
    162 uniform bool correct_gamma;
    163 
    164 void main()
    165 {
    166     color = texture( render_tex, UV ).xyz;
    167 }
    168 )";
        




cudaGraphicsMapResources : Map graphics resources for access by CUDA. 
-----------------------------------------------------------------------

* "map passes baton from OpenGL to CUDA"

* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html

Maps the count graphics resources in resources for access by CUDA.

The resources in resources may be accessed by CUDA until they are unmapped. The
graphics API from which resources were registered should not access any
resources while they are mapped by CUDA. If an application does so, the results
are undefined.

This function provides the synchronization guarantee that any graphics calls
issued before cudaGraphicsMapResources() will complete before any subsequent
CUDA work issued in stream begins. 


cudaGraphicsResourceGetMappedPointer
-----------------------------------------

* Get an device pointer through which to access a mapped graphics resource. 






