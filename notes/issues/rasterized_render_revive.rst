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


   




