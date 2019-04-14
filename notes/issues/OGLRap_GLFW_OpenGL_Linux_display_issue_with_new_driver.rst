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

Linux renders as above, but suspiciously reports OpenGL 3.2.0::

    [blyth@localhost UseInstance]$ LD_LIBRARY_PATH=~/local/opticks/lib64 UseInstanceTest
    Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    Frame::gl_init_window OpenGL version supported 3.2.0 NVIDIA 418.56
    [blyth@localhost UseInstance]$ LD_LIBRARY_PATH=~/local/opticks/lib64 OneTriangleTest
    Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    Frame::gl_init_window OpenGL version supported 3.2.0 NVIDIA 418.56
    [blyth@localhost UseInstance]$ 


Modify the hinting for different platforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    #if defined __APPLE__
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3); 
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
        glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // this incantation gives
        //    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
        //    OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

    #elif defined _MSC_VER
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1); 
     
    #elif __linux
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1); 

        //  executing UseInstanceTest
        // Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
        // Frame::gl_init_window OpenGL version supported 4.1.0 NVIDIA 418.56
    #endif



This succeeds to get the needed version reported on Linux, but still have the issue::

    2019-04-13 17:16:03.759 ERROR [299660] [Frame::initContext@329] Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    2019-04-13 17:16:03.759 ERROR [299660] [Frame::initContext@330] Frame::gl_init_window OpenGL version supported 4.1.0 NVIDIA 418.56


Hmm : perhaps the driver update and Linux kernel update conspired to push to newer OpenGL version ?
-----------------------------------------------------------------------------------------------------------

* actually it seems the converse, the GLFW hinting incantation needs to be modified to get at least 
  OpenGL 4.1 on Linux : in current form get 3.2.0  

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






OKTest again, shows 
-----------------------------------

On Linux, the OpenGL version reported is coming out as 3.2.0::

    2019-04-13 16:50:16.923 ERROR [258331] [Frame::initContext@306] Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    2019-04-13 16:50:16.923 ERROR [258331] [Frame::initContext@307] Frame::gl_init_window OpenGL version supported 3.2.0 NVIDIA 418.56

But on Darwin OpenGL 4.1 is reported::

    2019-04-13 16:54:58.139 ERROR [7478368] [Frame::initContext@306] Frame::gl_init_window Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    2019-04-13 16:54:58.139 ERROR [7478368] [Frame::initContext@307] Frame::gl_init_window OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

The OGLRap shaders aint going to work with 3.2.0


This looks smoky, seems that with the new driver + GLX the GLFW hinting
incantation needs to be changed to pick up a new enough OpenGL 4.1 minimum.

But after fixing this the render is still mangled. 


About OpenGL core profile
-----------------------------

* https://retokoradi.com/2014/03/30/opengl-transition-to-core-profile/


AxisAppCheck aka UseOGLRap
----------------------------

Perusing the oglrap Renderers they are compilcated : so focus on AxisAppCheck aka UseOGLRap
which is also afflicted.

macOS: small red, gree, blue axis
Linux: starts black, a blue line appears on mousing around


Composition::update setups the axis data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    0146   m_light_position(0,0,0,1),   // avoid it ever being undefined
    0147   m_light_direction(0,0,1,0),
    ...
    0186   m_axis_x(1000.f,    0.f,    0.f, 0.f),
     187   m_axis_y(0.f   , 1000.f,    0.f, 0.f),
     188   m_axis_z(0.f   ,    0.f, 1000.f, 0.f),
     189   m_axis_x_color(1.f,0.f,0.f,1.f),
     190   m_axis_y_color(0.f,1.f,0.f,1.f),
     191   m_axis_z_color(0.f,0.f,1.f,1.f),
    ...
    1761     m_axis_data->setQuad(m_light_position, 0,0 );
    1762     m_axis_data->setQuad(m_axis_x        , 0,1 );
    1763     m_axis_data->setQuad(m_axis_x_color  , 0,2 );
    1764 
    1765     m_axis_data->setQuad(m_light_position, 1,0 );
    1766     m_axis_data->setQuad(m_axis_y        , 1,1 );
    1767     m_axis_data->setQuad(m_axis_y_color  , 1,2 );
    1768 
    1769     m_axis_data->setQuad(m_light_position, 2,0 );
    1770     m_axis_data->setQuad(m_axis_z        , 2,1 );
    1771     m_axis_data->setQuad(m_axis_z_color  , 2,2 );
    1772 



Debugging Refs : Chasing the INVALID_ENUM 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://learnopengl.com/In-Practice/Debugging

One important thing left to mention is that GLEW has a long-existing bug where
calling glewInit() always sets the GL_INVALID_ENUM error flag and thus the
first glGetError will always return an error code which can throw you
completely off guard. To fix this it's advised to simply call glGetError after
glewInit to clear the flag: 

::

    glewInit();
    glGetError();


Avoid avoiding this problem, succeeded to prosecute the invalid enum on macOS
(it was GL_POLYGON_STIPPLE) and just removed the not working and not needed stipple method.






