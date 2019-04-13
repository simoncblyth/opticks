OGLRap_GLFW_OpenGL_Linux_display_issue_with_new_driver
=========================================================

Issue : Linux NVIDIA driver 418.56 glfw-3.2.1 from yum
---------------------------------------------------------------

Launch::

   OKTest 


* Window pops up, pressing Q twice to makes the geometry 
  appear in a mangled form : only along a narrow horizonal band. 

* Pressing R for rotation makes the narrow band of color get bigger
  and fills the screen with triangles emanating from a point on screen : as
  if a very wrong projection is used. 

* Pressing H for home gets back to the narrow band. Subsequently 
  any mouse movement (without any GUI keys enabled) looses the band 
  replaced with a near vertical dotted line. 

* Pressing G for GUI brings it up, and it displays and works normally
  on top of the bizarre geometry render



* same code running on macOS with GLFW 3.1.1 and NVIDIA Web Driver 387.10.10.10.40.105
  displays normally 

* pressing the Camera::Summary button on Linux and macOS shows the matrices
  agree as well as might aspect from the somwwhat different aspect ratios



Axis App Check
----------------

Try the axis app check::

   cd ~/opticks/examples/UseOGLRap
   ./go.sh

  
* macOS: black window pops up with small red-green-blue axis lines in the center, 
  these rotate in response to mouse movement after pressing R 

* Linux: black window starts empty, a single blue line can be made to appear by
  pressing R and moving mouse around 



UseOpticksGLFW : Primordial minimal GLFW OpenGL (no shaders)
---------------------------------------------------------------

* works as expected on macOS and Linux, showing the colorful rotating triangle


UseOpticksGLFWShaders : Need a minimal GLFW example with shaders 
-------------------------------------------------------------------

Start from the GLFW example https://www.glfw.org/docs/latest/quick.html#quick_example

* modify it to use GLEW and GLM.

SEGV at glGenBuffers, when omitted to glewInit() following window creation::

    gdb) bt
    #0  0x0000000000000000 in ?? ()
    #1  0x0000000000401276 in main () at /home/blyth/opticks/examples/UseOpticksGLFWShader/UseOpticksGLFWShader.cc:73
    (gdb) f 1
    #1  0x0000000000401276 in main () at /home/blyth/opticks/examples/UseOpticksGLFWShader/UseOpticksGLFWShader.cc:73
    73      glGenBuffers(1, &vertex_buffer);
    (gdb) 


With the hinting::

     65     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
     66     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
     67   /*
     68     By default, the OpenGL context GLFW creates may have any version. You can
     69     require a minimum OpenGL version by setting the GLFW_CONTEXT_VERSION_MAJOR and
     70     GLFW_CONTEXT_VERSION_MINOR hints before creation. If the required minimum
     71     version is not supported on the machine, context (and window) creation fails.
     72   */
     73     window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);


Linux succeeds to draw a triangle using a shader::

    executing UseOpticksGLFWShader
     renderer TITAN RTX/PCIe/SSE2 
     version 4.6.0 NVIDIA 418.56 

Darwin also ::

    executing UseOpticksGLFWShader
     renderer NVIDIA GeForce GT 750M OpenGL Engine 
     version 2.1 NVIDIA-10.33.0 387.10.10.10.40.105 

Changing hinting to::

     70     glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
     71     glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
     72     glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
     73     glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
     74 

Darwin fails to draw the triangle, just a black window::

    executing UseOpticksGLFWShader
     renderer NVIDIA GeForce GT 750M OpenGL Engine 
     version 4.1 NVIDIA-10.33.0 387.10.10.10.40.105 


This example does not use vao (vertex-attribute-object) so 
it is not really modern OpenGL.


UseInstance : shaders, vao, glDrawArraysInstanced
---------------------------------------------------

Bring env- instance- ~/env/graphics/opengl/instance/ into 
Opticks CMake environment.

Darwin : renders a diagonal line of blue instanced triangles::

    epsilon:UseInstance blyth$ DYLD_LIBRARY_PATH=$LOCAL_BASE/opticks/lib UseInstanceTest
    Frame::gl_init_window Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    Frame::gl_init_window OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

Darwin : renders a window filling blue triangle::

    epsilon:UseInstance blyth$ DYLD_LIBRARY_PATH=$LOCAL_BASE/opticks/lib OneTriangleTest
    Frame::gl_init_window Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    Frame::gl_init_window OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105



Hmm : perhaps the driver update and Linux kernel update conspired to push to newer OpenGL version ?
-----------------------------------------------------------------------------------------------------------

* do my shaders need some version spec ? 

::

    [blyth@localhost issues]$ glxinfo | grep NVIDIA
    server glx vendor string: NVIDIA Corporation
    client glx vendor string: NVIDIA Corporation
    OpenGL vendor string: NVIDIA Corporation
    OpenGL core profile version string: 4.6.0 NVIDIA 418.56
    OpenGL core profile shading language version string: 4.60 NVIDIA
    OpenGL version string: 4.6.0 NVIDIA 418.56
    OpenGL shading language version string: 4.60 NVIDIA
    OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 418.56
    [blyth@localhost issues]$ 


    [blyth@localhost issues]$ glxinfo | grep version
    server glx version string: 1.4
    client glx version string: 1.4
    GLX version: 1.4
    OpenGL core profile version string: 4.6.0 NVIDIA 418.56
    OpenGL core profile shading language version string: 4.60 NVIDIA
    OpenGL version string: 4.6.0 NVIDIA 418.56
    OpenGL shading language version string: 4.60 NVIDIA
    OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 418.56
    OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
        GL_EXT_shader_implicit_conversions, GL_EXT_shader_integer_mix, 
    [blyth@localhost issues]$ 





