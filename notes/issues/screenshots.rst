Screenshots
============

Objective
-----------

On macOS its trivial to make manual screenshots (press shift-cmd-4 and drag out a marquee).
With Linux doing there are many apps that can do this.

For uniformity and automation it would be good from Opticks to be able to do this from OpenGL 
level too.


Examples Exploring how to implement pixel downloading and saving to PPM
--------------------------------------------------------------------------

examples/UseOpticksGLFWSnap
    add PPM snapshots to the minimal GLFW triangle example using Pix struct 

examples/UseOpticksGLFWSPPM
    move methods of Pix into sysrap/SPPM for reusability, demonstrate usage here


OptiX level : capturing raytrace renders
-------------------------------------------

* see oxrap/OContext snap

OpenGL level : based on glReadPixels
-------------------------------------

Did this many moons ago with the old python daeview, in env repo.

::

    epsilon:env blyth$ find . -type f  -exec grep -H glReadPixels {} \; 
    ./pycuda/pycuda_pyopengl_interop/image_processor.py:import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
    ./pycuda/pycuda_pyopengl_interop/pixel_buffer.py:import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
    ./pycuda/pycuda_pyopengl_interop/pixel_buffer.py:        * http://pyopengl.sourceforge.net/documentation/manual-3.0/glReadPixels.html
    ./pycuda/pycuda_pyopengl_interop/pixel_buffer.py:        * https://www.opengl.org/discussion_boards/showthread.php/165780-PBO-glReadPixels-not-so-fast
    ./pycuda/pycuda_pyopengl_interop/pixel_buffer.py:        rawgl.glReadPixels(
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop_structured.py:#import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:        rawgl.glReadPixels(
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:    AttributeError: 'module' object has no attribute 'glReadPixels'
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:    import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   # only GL_1_1 has the glReadPixels symbol
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:    rawgl.glReadPixels(
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:    import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   # only GL_1_1 has the glReadPixels symbol
    ./pycuda/pycuda_pyopengl_interop/examples/GlInterop.py:    rawgl.glReadPixels(
    ./externals/optix.bash:    // sets pixel storage modes that affect the operation of subsequent glReadPixels 
    ./optix/optix.bash:    // sets pixel storage modes that affect the operation of subsequent glReadPixels 
    Binary file ./_build/doctrees/optix/optix.doctree matches
    Binary file ./_build/doctrees/graphics/glumpy/glumpy.doctree matches
    ./geant4/geometry/collada/g4daeview/daecamera.py:        CAUTION: y flip in pixel numbering compared to glReadPixels
    ./geant4/geometry/collada/g4daeview/daecamera.py:        * https://www.opengl.org/sdk/docs/man2/xhtml/glReadPixels.xml
    ./geant4/geometry/collada/g4daeview/daecamera.py:        glReadPixels numbers pixels with lower left corner at (x+i,y+j )
    ./geant4/geometry/collada/g4daeview/daeframehandler.py:        pixels = gl.glReadPixelsf(x, y, 1, 1, gl.GL_DEPTH_COMPONENT ) # width,height 1,1  
    ./geant4/geometry/collada/g4daeview/daeframehandler.py:        data = gl.glReadPixels (x,y,w,h, format_, type_)
    ./graphics/egl/myfirst.cc:        glReadPixels(0,0,renderBufferWidth,renderBufferHeight,GL_RGBA, GL_UNSIGNED_BYTE, data2);
    ./graphics/glumpy/glumpy.bash:My workaround is to avoid FBO and simply glReadPixels from the framebuffer::
    ./oglrap/Frame.cc:    glReadPixels(x, y, width, height, format, type, &depth ); 
    epsilon:env blyth$ 



