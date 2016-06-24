# === func-gen- : graphics/oglrap/oglrap fgp graphics/oglrap/oglrap.bash fgn oglrap fgh graphics/oglrap
oglrap-rel(){      echo graphics/oglrap ; }
oglrap-src(){      echo graphics/oglrap/oglrap.bash ; }
oglrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglrap-src)} ; }
oglrap-vi(){       vi $(oglrap-source) ; }
oglrap-usage(){ cat << EOU

Featherweight OpenGL wrapper
==============================

Just a few utility classes to make modern OpenGL 3, 4 
easier to use.

Originally was thinking could keep GLFW3 out of oglrap
keeping pure OpenGL, but that has proved difficult.

GLFW3 and GLEW are responsible for furnishing the 
OpenGL headers, but wanted to avoid directly using 
GLFW3 inside oglrap-



Switch Ortho to Frustum without "loosing" view
-----------------------------------------------

* :google:`opengl switch frustum to ortho` 

  * http://www.songho.ca/opengl/gl_transform.html
  * http://compgroups.net/comp.graphics.api.opengl/perspective-orthographic-switch/171949

Perhaps arranging the perspective frustum dimensions at  (near+far)/2 
to line up with othographic box will avoid the sudden change of viewpoint.

How do other 3D open source projects do the switch  ? 
Blender source hard to follow. TODO: take a look at MeshLab

* https://github.com/dfelinto/blender/search?utf8=✓&q=ortho_scale


Better Shader Handling ?
--------------------------

* want less tedium when adding/setting uniforms/attributes 
 
  * https://github.com/mmmovania/opengl33_dev_cookbook_2013/blob/master/Chapter2/src/GLSLShader.cpp

  * https://github.com/OpenGLInsights/OpenGLInsightsCode


Vertex Attributes
-------------------

* https://www.opengl.org/wiki/Vertex_Specification_Best_Practices

what is best way to hookup shader attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/4635913/explicit-vs-automatic-attribute-location-binding-for-opengl-shaders
* https://www.packtpub.com/books/content/tips-and-tricks-getting-started-opengl-and-glsl-40
* https://www.youtube.com/watch?v=mL6BvXVtd9Y


* glBindAttribLocation() before linking to explicitly define an attribute location.
* glGetAttribLocation() after linking to obtain an automatically assigned attribute location.
* layout(location=0) in vec4 position;   explicitly specify a location using layout

  * this way you can skip the Bind or Get and just assume the layout in the GLSL, but that
    duplicates a location numbers in host and device  : which is fragile

  * using layout AND Get is explicit, and allows to avoid the repetition of a number, instead
    repeat the attribute name which seems a lot less fragile

  * caution regards layout, mat4 takes 4 layout slots     


Currently using enum values in code that duplicate layout numbers in vert.glsl
WHICH IS VERY FRAGILE::

    delta:gl blyth$ grep layout */vert.glsl
    nrm/vert.glsl:layout(location = 0) in vec3 vertex_position;
    nrm/vert.glsl:layout(location = 1) in vec3 vertex_colour;
    nrm/vert.glsl:layout(location = 2) in vec3 vertex_normal;
    nrm/vert.glsl:layout(location = 3) in vec2 vertex_texcoord;
    pos/vert.glsl:layout(location = 10) in vec3 vertex_position;
    tex/vert.glsl:layout(location = 0) in vec3 vertex_position;
    tex/vert.glsl:layout(location = 1) in vec3 vertex_colour;
    tex/vert.glsl:layout(location = 2) in vec3 vertex_normal;
    tex/vert.glsl:layout(location = 3) in vec2 vertex_texcoord;
    delta:gl blyth$ 


Better to compile and link shader first thing, then can grab locations
and use those being care with which buffers are acitive::

    int weightPosition = glGetAttribLocation(programID, "blendWeights");
    // does this always give the "layout" value from the shader ? 

    glVertexAttribPointer(weightPosition, 4, GL_FLOAT, GL_FALSE, sizeof(TVertex_VNTWI), info->weightOffset);
    glEnableVertexAttribArray(weightPosition);


Idea for step animation using line to point geometry shader alone
--------------------------------------------------------------------

1. start with zeroed out step buffer destined to contain 
   (position,time) "post" quads for each step of the photons 
   up to a fixed max steps

   * need to zero out, or use -1. to enable distinguishing empty slots 
     (time value needs to be arranged to never be 0. or negative) 

   * even when generating photons on GPU the total number of photons 
     is known ahead of time and max_steps is an input

2. OptiX launch using OpenGL interop VBO for the step buffer 

   * generates photons filling first slot with position_time

   * subsequent propagation fills out subsequent slots up to
     maxsteps, for any one photon the slots are contiguous

   * could cheat by repeated overwriting the maximal recorded slot 
     in order to capture the more interesting last step 
     (up to some defined maximum steps to trace, 
     which can be a lot larger than maximum steps to record)
 
   * many blanks will remain in the buffer from early stops
     
     * the simple splayed buffer structure is to avoid 
       concurrency complications, as every photon owns its own chunk of buffer 
      
     * dont really know any other way of doing this, even using atomics
       to serialize writing into a tight buffer would then have all photons
       steps interleaved, would need an additional structure to record 
       indices corresponding to each photon, but that has concurrency problem 
       too ?    
 
3. glDrawArrays(GL_LINES,...) on the step buffer using Geometry 
   shader that takes lines as input and emits points with a
   uniform time float input.

   Geometry shader has access to two (position,time) vertices
   comparing times with the uniform can decide to either:

   * cull when step times do not straddle the input time 
   * interpolate positions between the steps based on their
     times and the input time 


Note that there is no need to distinguish between photons 
in the draw call, the entire step buffer can be interpreted 
as potential lines between steps to be interpolated.  
Probably should arrange to always have a gap between recorded 
steps in the buffer to avoid spurious interpolation 
between separate photons, although its unlikely that 
the step times would conspire to make this happen.

Formerly used a separate CUDA kernel to do the find relevant 
steps, interpolate between them and write into a top slot
allowing an OpenGL fixed stride draw.


TODO
-----

* genstep PDG code coloring ? maybe PDG code integer attribute
  (unlike ancient OpenGL, shaders should be able to handle integers now)
  If that proves difficult could pre-cook a color buffer

* interoptix with the genstep VBO, and porting cerenkov and scintillation 
  generation code to OptiX

  * split OptiXEngine into 

    * non-OpenGL OptiXCore 
    * separate OptiXRenderer

  Currently are basing off of OpenGL buffer objects, need to 
  arrange OptiX setup code to be agnostic in this regard 


DONE
-----

* visualizing generated photon initial positions

* add clipping planes to "nrm" shaders as check of 
  shader uniform handling and as need to clip 

* genstep viz using Rdr with VecNPY addressing 
  and p2l (point to line) geometry shader based on my ancient one



Windows Launching
--------------------

Informative::

    $ BookmarksTest.exe
    Segmentation fault

Many absentees::

    $ ldd $(which BookmarksTest.exe) | grep opticks
            libOpticksCore.dll => /usr/local/opticks/lib/libOpticksCore.dll (0x623c0000)
            libBCfg.dll => /usr/local/opticks/lib/libBCfg.dll (0x65180000)
            libBRegex.dll => /usr/local/opticks/lib/libBRegex.dll (0x6cbc0000)
            libNPY.dll => /usr/local/opticks/lib/libNPY.dll (0x1cd0000)




Classes
--------


Frame
       OpenGL context creation and window control 
Interactor
       GLFW event handling and passing off to Camera, Trackball, View, Clipper etc..


Composition
       matrix calculations based on the Camera, Trackball, View and Clipper constituents
Camera
       near/far/...
Trackball
       quaternion calculation of perturbing rotations, and translations 
View  
       eye/look/up  
Clipper
       clipping plane control


Geometry
       high level control of geometry loading 

Rdr
       Specialization of RendererBase used for 
       event data rendering, ie not geometry,
       with tags: 

       pos : used directly from GGeoView main for 
             visualizing VecNPY event data

Renderer 
       Specialization of RendererBase used for 
       geometrical rendering and also OptiX texture/PBO 
       presentation.
 
       nrm : normal shader used directly from GGeoView main
       tex : quad texture used by OptiXEngine   

RendererBase
       handles shader program access, compilation and linking 
       using Prog and Shdr classes
Prog 
       representation of shader program pipeline 
       comprised of one for more shaders
Shdr
       single shader


Texture
       (misnamed : Quad might be better)
       GMesh subclass for a quadrangle, used for rendering 
       OptiX generated PBOs Pixel Buffer Objects via OpenGL textures 

Demo
       GMesh subclass representing a single triangle geometry  


CameraCfg
CompositionCfg
InteractorCfg
RendererCfg
TrackBallCfg
ViewCfg
ClipperCfg
      configuration connector classes enabling commandline or live 
      config of objects





EOU
}

oglrap-env(){      elocal- ; opticks- ; }

oglrap-sdir(){ echo $(env-home)/graphics/oglrap ; }
oglrap-tdir(){ echo $(env-home)/graphics/oglrap/tests ; }
oglrap-idir(){ echo $(opticks-idir) ; }
oglrap-bdir(){ echo $(opticks-bdir)/$(oglrap-rel) ; }

oglrap-bindir(){ echo $(oglrap-idir)/bin ; }

oglrap-cd(){   cd $(oglrap-sdir); }
oglrap-scd(){  cd $(oglrap-sdir); }
oglrap-tcd(){  cd $(oglrap-tdir); }
oglrap-icd(){  cd $(oglrap-idir); }
oglrap-bcd(){  cd $(oglrap-bdir); }


oglrap-name(){ echo OGLRap ; }
oglrap-tag(){  echo OGLRAP ; }

oglrap-wipe(){ local bdir=$(oglrap-bdir) ; rm -rf $bdir ;  } 

oglrap--(){        opticks--     $(oglrap-bdir) ; }
oglrap-ctest(){    opticks-ctest $(oglrap-bdir) $* ; }
oglrap-genproj(){  oglrap-scd ; opticks-genproj $(oglrap-name) $(oglrap-tag) ; }
oglrap-gentest(){  oglrap-tcd ; opticks-gentest ${1:-Scene} $(oglrap-tag) ; }
oglrap-txt(){     vi $(oglrap-sdir)/CMakeLists.txt $(oglrap-tdir)/CMakeLists.txt ; }


oglrap-export()
{
   export SHADER_DIR=$(oglrap-sdir)/gl
} 

oglrap-run(){ 
   local bin=$(oglrap-bindir)/OGLRapTest
   oglrap-export
   $bin $*
}

oglrap-frametest()
{
   local path=${1:-/tmp/teapot.ppm}
   shift
   local bin=$(oglrap-bindir)/FrameTest
   oglrap-export
   $LLDB $bin $path $*
}


oglrap-loadtest(){
   local bin=$(oglrap-bindir)/GLoaderTest
   ggeoview-
   ggeoview-export
   $LLDB $bin $*  
}


oglrap-frametest-lldb()
{
   LLDB=lldb oglrap-frametest $*
}

oglrap-progtest()
{
   SHADER_DIR=~/env/graphics/ggeoview/gl $(oglrap-bindir)/ProgTest
}
