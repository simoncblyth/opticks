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





Arcball
----------

* https://oguz81.github.io/ArcballCamera/
* https://github.com/oguz81/ArcballCamera


* Properties of Quaternions


Arcball with quaternions
--------------------------

* https://research.cs.wisc.edu/graphics/Courses/559-f2001/Examples/Gl3D/arcball-gems.pdf

* ~/opticks_refs/ken_shoemake_arcball_rotation_control_gem.pdf

* https://github.com/Twinklebear/arcball-cpp




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


   




