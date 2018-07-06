# === func-gen- : oglrap/oglrap fgp oglrap/oglrap.bash fgn oglrap fgh oglrap
oglrap-src(){      echo oglrap/oglrap.bash ; }
oglrap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oglrap-src)} ; }
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


Instancing Refs
-----------------

* https://learnopengl.com/#!Advanced-OpenGL/Instancing
* http://sol.gfxile.net/instancing.html performance comparisons
* https://www.khronos.org/opengl/wiki/Vertex_Rendering
* http://ogldev.atspace.co.uk/www/tutorial33/tutorial33.html 




CentOS7 GLEQ
---------------

::

	[ 71%] Building CXX object CMakeFiles/OGLRap.dir/OpticksViz.cc.o
	In file included from /home/blyth/opticks/oglrap/GLEQ.hh:7:0,
		 from /home/blyth/opticks/oglrap/Frame.hh:9,
		 from /home/blyth/opticks/oglrap/OpticksViz.cc:43:
	/home/blyth/opticks/oglrap/gleq.h:37:6: warning: #warning "This version of GLEQ does not support events added after GLFW 3.1" [-Wcpp]
	#warning "This version of GLEQ does not support events added after GLFW 3.1"
	^


Centos7 Imgui linking
----------------------


::

        om-install

	[ 74%] Linking CXX shared library libOGLRap.so
	[ 74%] Built target OGLRap
	Scanning dependencies of target ProgTest
	[ 76%] Building CXX object tests/CMakeFiles/ProgTest.dir/ProgTest.cc.o
	[ 79%] Linking CXX executable ProgTest
	../libOGLRap.so: undefined reference to `ImGui::SliderFloat(char const*, float*, float, float, char const*, float)'
	../libOGLRap.so: undefined reference to `ImGui::Checkbox(char const*, bool*)'
	../libOGLRap.so: undefined reference to `ImGui::ShowTestWindow(bool*)'
	../libOGLRap.so: undefined reference to `ImGui::SliderInt(char const*, int*, int, int, char const*)'
	../libOGLRap.so: undefined reference to `ImGui::Text(char const*, ...)'
	../libOGLRap.so: undefined reference to `ImGui::Render()'
	../libOGLRap.so: undefined reference to `ImGui::TextColored(ImVec4 const&, char const*, ...)'
	../libOGLRap.so: undefined reference to `ImGui::GetIO()'
	../libOGLRap.so: undefined reference to `ImGui::PushItemWidth(float)'


opticks/examples/UseImGui::

	[ 50%] Linking CXX executable UseImGui
	/home/blyth/local/opticks/externals/lib/libImGui.so: undefined reference to `glfwSetScrollCallback'
	/home/blyth/local/opticks/externals/lib/libImGui.so: undefined reference to `glfwGetTime'
	/home/blyth/local/opticks/externals/lib/libImGui.so: undefined reference to `glfwSetKeyCallback'
	/home/blyth/local/opticks/externals/lib/libImGui.so: undefined reference to `glfwSetClipboardString'
	/home/blyth/local/opticks/externals/lib/libImGui.so: undefined reference to `glfwGetWindowSize'





FrameTest cleanup error that hung system, forcing reboot
------------------------------------------------------------


* suspect cause of hang was due to another OpenGL context in limbo in another debugger session

  * nope it happened again from a clean environment
  * try to remember to cleanup windows and other debug sessions before getting graphical

::

    2017-08-18 16:36:25.817 DEBUG [510048] [Frame::setSize@194] Frame::setSize  width 524 height 585 coord2pixel 2
    2017-08-18 16:36:28.595 INFO  [510048] [Frame::key_pressed@703] Frame::key_pressed escape
    FrameTest(95194,0x7fff7263c310) malloc: *** error for object 0x1013df960: pointer being freed was not allocated
    *** set a breakpoint in malloc_error_break to debug
    Process 95194 stopped
    * thread #1: tid = 0x7c860, 0x00007fff8c2db866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8c2db866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8c2db866:  jae    0x7fff8c2db870            ; __pthread_kill + 20
       0x7fff8c2db868:  movq   %rax, %rdi
       0x7fff8c2db86b:  jmp    0x7fff8c2d8175            ; cerror_nocancel
       0x7fff8c2db870:  retq   
    (lldb) bt
    * thread #1: tid = 0x7c860, 0x00007fff8c2db866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8c2db866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8397835c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8a6c8b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8410007f libsystem_malloc.dylib`free + 411
        frame #4: 0x000000010043bd07 libGGeo.dylib`GMesh::deallocate(this=0x00007fff5fbfe850) + 55 at GMesh.cc:194
        frame #5: 0x000000010043bf12 libGGeo.dylib`GMesh::~GMesh(this=0x00007fff5fbfe850) + 34 at GMesh.cc:215
        frame #6: 0x0000000100010cc5 FrameTest`Texture::~Texture(this=0x00007fff5fbfe850) + 21 at Texture.hh:26
        frame #7: 0x0000000100007425 FrameTest`Texture::~Texture(this=0x00007fff5fbfe850) + 21 at Texture.hh:26
        frame #8: 0x0000000100006c42 FrameTest`main(argc=4, argv=0x00007fff5fbfed98) + 2434 at FrameTest.cc:100
        frame #9: 0x00007fff8774e5fd libdyld.dylib`start + 1
        frame #10: 0x00007fff8774e5fd libdyld.dylib`start + 1
    (lldb) 
      [Restored]:
    Last login: Fri Aug 18 16:42:14 on ttys003
    simon:oglrap blyth$ 



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

oglrap-env(){      olocal- ; opticks- ; }

oglrap-sdir(){ echo $(opticks-home)/oglrap ; }
oglrap-tdir(){ echo $(opticks-home)/oglrap/tests ; }
oglrap-idir(){ echo $(opticks-idir) ; }
oglrap-bdir(){ echo $(opticks-bdir)/oglrap ; }

oglrap-bindir(){ echo $(oglrap-idir)/bin ; }

oglrap-c(){    cd $(oglrap-sdir); }
oglrap-cd(){   cd $(oglrap-sdir); }
oglrap-scd(){  cd $(oglrap-sdir); }
oglrap-tcd(){  cd $(oglrap-tdir); }
oglrap-icd(){  cd $(oglrap-idir); }
oglrap-bcd(){  cd $(oglrap-bdir); }


oglrap-name(){ echo OGLRap ; }
oglrap-tag(){  echo OGLRAP ; }

oglrap-wipe(){ local bdir=$(oglrap-bdir) ; rm -rf $bdir ;  } 


oglrap-apihh(){  echo $(oglrap-sdir)/$(oglrap-tag)_API_EXPORT.hh ; }
oglrap---(){     touch $(oglrap-apihh) ; oglrap--  ; }


oglrap--(){        opticks--     $(oglrap-bdir) ; }
oglrap-t(){       opticks-t $(oglrap-bdir) $* ; }
oglrap-clean(){   opticks-make- $(oglrap-bdir) clean ; }
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
   #local path=${1:-/tmp/teapot.ppm}
   local path=${1:-/opt/local/lib/tk8.6/demos/images/teapot.ppm}
   shift
   local bin=FrameTest
   oglrap-export
   lldb $bin $path $* -- --OGLRAP trace
}


oglrap-progtest()
{
   SHADER_DIR=~/env/graphics/ggeoview/gl $(oglrap-bindir)/ProgTest
}

oglrap-instcull()
{
    oglrap-c

    op --j1707 \
       --gltf 3 \
       --tracer \
       --instcull --lod 1 --lodconfig "levels=3,verbosity=2,instanced_lodify_onload=1" \
       --debugger \
       --target 12 --eye 0.7,0.7,0. \
       --rendermode +bb0,+in1,+in2,+in3,-global

}

oglrap-link-issue()
{

cd /home/blyth/local/opticks/build/oglrap/tests 
/usr/bin/c++ \
       -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow -g  \
       CMakeFiles/ProgTest.dir/ProgTest.cc.o  -o ProgTest \
        ../libOGLRap.so  \
        /home/blyth/local/opticks/externals/lib/libImGui.so  \
        /home/blyth/local/opticks/lib64/libSysRap.so \


    cat << EOL > /dev/null
       -fvisibility=hidden -fvisibility-inlines-hidden \
        -Wl,-rpath,/home/blyth/local/opticks/build/oglrap:/home/blyth/local/opticks/externals/lib64:/home/blyth/local/opticks/externals/lib:/home/blyth/local/opticks/lib64: \
        -lGL 
        /usr/lib64/libglfw.so  \
        /home/blyth/local/opticks/externals/lib64/libGLEW.so  \

     /usr/lib64/libboost_program_options-mt.so \
     /usr/lib64/libboost_filesystem-mt.so \
     /usr/lib64/libboost_system-mt.so \
     /usr/lib64/libboost_regex-mt.so \
       -lssl -lcrypto \

EOL

}

oglrap-link-issue-notes(){ cat << EON

::

	[blyth@localhost tests]$ oglrap-link-issue
	../libOGLRap.so: undefined reference to `ImGui::SliderFloat(char const*, float*, float, float, char const*, float)'
	../libOGLRap.so: undefined reference to `ImGui::Checkbox(char const*, bool*)'
	../libOGLRap.so: undefined reference to `ImGui::ShowTestWindow(bool*)'
	../libOGLRap.so: undefined reference to `ImGui::SliderInt(char const*, int*, int, int, char const*)'
	../libOGLRap.so: undefined reference to `ImGui::Text(char const*, ...)'
	../libOGLRap.so: undefined reference to `ImGui::Render()'
	../libOGLRap.so: undefined reference to `ImGui::TextColored(ImVec4 const&, char const*, ...)'
	../libOGLRap.so: undefined reference to `ImGui::GetIO()'
	../libOGLRap.so: undefined reference to `ImGui::PushItemWidth(float)'
	../libOGLRap.so: undefined reference to `ImGui::End()'
	../libOGLRap.so: undefined reference to `ImGui_ImplGlfwGL3_NewFrame()'
	../libOGLRap.so: undefined reference to `ImGui::SetWindowFontScale(float)'
	../libOGLRap.so: undefined reference to `ImGui::RadioButton(char const*, int*, int)'
	../libOGLRap.so: undefined reference to `ImGui::Button(char const*, ImVec2 const&)'
	../libOGLRap.so: undefined reference to `ImGui::SliderFloat3(char const*, float*, float, float, char const*, float)'
	../libOGLRap.so: undefined reference to `ImGui::Begin(char const*, bool*, ImVec2 const&, float, int)'
	../libOGLRap.so: undefined reference to `ImGui_ImplGlfwGL3_Shutdown()'
	../libOGLRap.so: undefined reference to `ImGui::SetNextWindowPos(ImVec2 const&, int)'
	../libOGLRap.so: undefined reference to `ImGui_ImplGlfwGL3_Init(GLFWwindow*, bool)'
	../libOGLRap.so: undefined reference to `ImGui::Spacing()'
	../libOGLRap.so: undefined reference to `ImGui::CollapsingHeader(char const*, char const*, bool, bool)'
	../libOGLRap.so: undefined reference to `ImGui::SameLine(float, float)'
	collect2: error: ld returned 1 exit status

	[blyth@localhost tests]$ nm ../libOGLRap.so | c++filt | grep ImGui
			 U ImGui_ImplGlfwGL3_Init(GLFWwindow*, bool)
			 U ImGui_ImplGlfwGL3_NewFrame()
			 U ImGui_ImplGlfwGL3_Shutdown()
			 U ImGui::RadioButton(char const*, int*, int)
			 U ImGui::SliderFloat(char const*, float*, float, float, char const*, float)
			 U ImGui::TextColored(ImVec4 const&, char const*, ...)
			 U ImGui::SliderFloat3(char const*, float*, float, float, char const*, float)
			 U ImGui::PushItemWidth(float)
			 U ImGui::ShowTestWindow(bool*)
			 U ImGui::CollapsingHeader(char const*, char const*, bool, bool)
			 U ImGui::SetNextWindowPos(ImVec2 const&, int)
			 U ImGui::SetWindowFontScale(float)
			 U ImGui::End()
			 U ImGui::Text(char const*, ...)
			 U ImGui::Begin(char const*, bool*, ImVec2 const&, float, int)
			 U ImGui::GetIO()
			 U ImGui::Button(char const*, ImVec2 const&)
			 U ImGui::Render()
			 U ImGui::Spacing()
			 U ImGui::Checkbox(char const*, bool*)
			 U ImGui::SameLine(float, float)
			 U ImGui::SliderInt(char const*, int*, int, int, char const*)
	[blyth@localhost tests]$ 


	[blyth@localhost tests]$ nm /home/blyth/local/opticks/externals/lib/libImGui.so | c++filt | grep SliderFloat 
	0000000000019f41 t ImGui::SliderFloat(char const*, float*, float, float, char const*, float)
	000000000001aba8 t ImGui::SliderFloat2(char const*, float*, float, float, char const*, float)
	000000000001ac07 t ImGui::SliderFloat3(char const*, float*, float, float, char const*, float)
	000000000001ac66 t ImGui::SliderFloat4(char const*, float*, float, float, char const*, float)
	000000000001aa5c t ImGui::SliderFloatN(char const*, float*, int, float, float, char const*, float)
	000000000001a47d t ImGui::VSliderFloat(char const*, ImVec2 const&, float*, float, float, char const*, float)

	[blyth@localhost tests]$ nm /home/blyth/local/opticks/externals/lib/libImGui.so | c++filt | grep Checkbox
	000000000001dc95 t ImGui::CheckboxFlags(char const*, unsigned int*, unsigned int)
	000000000001d513 t ImGui::Checkbox(char const*, bool*)



EON
}
