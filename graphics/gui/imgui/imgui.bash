# === func-gen- : graphics/gui/imgui/imgui fgp graphics/gui/imgui/imgui.bash fgn imgui fgh graphics/gui/imgui
imgui-src(){      echo graphics/gui/imgui/imgui.bash ; }
imgui-source(){   echo ${BASH_SOURCE:-$(env-home)/$(imgui-src)} ; }
imgui-vi(){       vi $(imgui-source) ; }
imgui-usage(){ cat << EOU

ImGUI (MIT) : Immediate Mode GUI
===================================

* https://github.com/ocornut/imgui

* includes an embedded console : developer-centric

These *imgui-* functions add cmake building to ImGui
that allows usage of env/cmake/Modules/FindImGui.cmake 
This is tested by imguitest-


Issues
-------

* keeps dropping a imgui.ini from the launch directory, how to control or change location ?

* need to find way to share input events between GLEQ and ImGui


Blurry Text
-----------

* https://github.com/ocornut/imgui

::

    I integrated ImGui in my engine and the text or lines are blurry..
    In your Render function, try translating your projection matrix by (0.5f,0.5f) or (0.375f,0.375f).


Discussion
-----------

* https://www.allegro.cc/forums/thread/615234


Summarizing an article on IMGUI
---------------------------------

* http://www.johno.se/book/imgui.html

GUIs traditionally duplicate some portion of application state 
and demand a synchronization so that state sloshes back and forth
between GUI and application.  

IMGUI eliminates the syncing by always passing the state...

* widgets no longer objects, become procedural method calls

* simplicity comes at expense of constantly resubmitting state 
  and redrawing the user interface at real-time rates. 

Thoughts
----------

* might be a good fit, i want a minimal GUI that is usually not shown,
  would like everything to be doable from console and over UDP messaging 


SEGV
-----

::

    (lldb) f 4
    frame #4: 0x0000000101a72253 libGGeoViewLib.dylib`App::renderGUI(this=0x00007fff5fbfed18) + 35 at App.cc:958
       955  void App::renderGUI()
       956  {
       957  #ifdef GUI_
    -> 958      m_gui->newframe();
       959      bool* show_gui_window = m_interactor->getGUIModeAddress();
       960      if(*show_gui_window)
       961      {
    (lldb) p m_gui
    (GUI *) $0 = 0x000000013e0f2bb0
    (lldb) f 3
    frame #3: 0x0000000101c6b4d1 libOGLRap.dylib`GUI::newframe(this=0x000000013e0f2bb0) + 17 at GUI.cc:76
       73   
       74   void GUI::newframe()
       75   {
    -> 76       ImGui_ImplGlfwGL3_NewFrame();
       77   }
       78   
       79   void GUI::choose( std::vector<std::pair<int, std::string> >* choices, bool* selection )
    (lldb) f 2
    frame #2: 0x00000001017c4c20 libImGui.dylib`ImGui_ImplGlfwGL3_NewFrame() + 32 at imgui_impl_glfw_gl3.cpp:325
       322  void ImGui_ImplGlfwGL3_NewFrame()
       323  {
       324      if (!g_FontTexture)
    -> 325          ImGui_ImplGlfwGL3_CreateDeviceObjects();
       326  
       327      ImGuiIO& io = ImGui::GetIO();
       328  
    (lldb) f 1
    frame #1: 0x00000001017c3f7d libImGui.dylib`ImGui_ImplGlfwGL3_CreateDeviceObjects() + 525 at imgui_impl_glfw_gl3.cpp:215
       212          "   Out_Color = Frag_Color * texture( Texture, Frag_UV.st);\n"
       213          "}\n";
       214  
    -> 215      g_ShaderHandle = glCreateProgram();
       216      g_VertHandle = glCreateShader(GL_VERTEX_SHADER);
       217      g_FragHandle = glCreateShader(GL_FRAGMENT_SHADER);
       218      glShaderSource(g_VertHandle, 1, &vertex_shader, 0);
    (lldb) f 0
    frame #0: 0x0000000000000000
    error: memory read failed for 0x0
    (lldb) 





FUNCTIONS
-----------

*imgui-copy*
     Creates imgui sub-folder and copies in the sources



Build opengl3 example
-----------------------

::

    simon:opengl3_example blyth$ diff Makefile.original Makefile
    32,33c32,33
    <   LIBS += -L/usr/local/lib
    <   LIBS += -lglfw3
    ---
    >   LIBS += -L$(GLFW_PREFIX)/lib
    >   LIBS += -lglfw.3
    35c35
    <   CXXFLAGS = -I../../ -I../libs/gl3w -I/usr/local/Cellar/glew/1.10.0/include -I/usr/local/include
    ---
    >   CXXFLAGS = -I../../ -I../libs/gl3w -I/usr/local/Cellar/glew/1.10.0/include -I/usr/local/include -I$(GLFW_PREFIX)/include
    simon:opengl3_example blyth$ 



April 2016 : imgui fix
-------------------------

::

    simon:imgui.build blyth$ grep gl3w.h /usr/local/opticks/externals/imgui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp
    #include <GL/gl3w.h>
    simon:imgui.build blyth$ perl -pi -e 's,gl3w.h,glew.h,' /usr/local/opticks/externals/imgui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp
    simon:imgui.build blyth$ grep gl3w.h /usr/local/opticks/externals/imgui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp




ggeoview integration example
------------------------------

imgui is not provided as a library, seems to be expected
to copy source into your tree

::

    ggeoview-
    ggeoview-cd
    imgui-
    imgui-copy     # copy sources into imgui folder

    cd imgui/examples/opengl3_example

   
Building ggeoview which uses GLEW gets tangled
------------------------------------------------

* https://github.com/ocornut/imgui/commit/3ed38f331347f4f01b425fde7059f5c3cc5227a6

The above commit looks to have replaced a contained 
glew with a hand-rolled gl3w.
When you have GLEW operational that is a reversion.

ImGui gl3w vs glew
-------------------

gl3w does the same thing as glew : need to use one or other


Event Sharing
---------------

imgui.cpp::

    after calling ImGui::NewFrame() you can read back 

    * 'io.WantCaptureMouse' and 'io.WantCaptureKeyboard' 

    to tell if ImGui wants to use your inputs. 
    If it does you can discard/hide the inputs from the rest of your application.


Hmm disabling "install_callbacks" in init does not prevent interaction with the GUI ?
And enables events to get through to my handling. How is this working ?


GUI photon selector, so can display only certain boundaries
------------------------------------------------------------

* Rdr needs to hookup an integer uniform to provide selected boundary code 
  to the shader : so can change color/visibility based on the selection


EOU
}

imgui-env(){      elocal- ; opticks- ;  }

imgui-edir(){ echo $(opticks-home)/graphics/gui/imgui ; }

imgui-oldbase(){ echo $(local-base)/env/graphics/gui ; }

imgui-base(){ echo $(opticks-prefix)/externals/imgui ; }
#imgui-base(){ echo $(imgui-oldbase) ; }

imgui-diff(){
  # diff --brief  $(imgui-oldbase) $(imgui-base)/imgui 

   diff /usr/local/env/graphics/gui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp \
       /usr/local/opticks/externals/imgui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp 

}


imgui-idir(){ echo $(imgui-base)/imgui.install ; }
imgui-bdir(){ echo $(imgui-base)/imgui.build   ; }
imgui-sdir(){ echo $(imgui-base)/imgui ; }
imgui-dir(){  echo $(imgui-base)/imgui ; }

imgui-ecd(){  cd $(imgui-edir); }
imgui-icd(){  cd $(imgui-idir); }
imgui-bcd(){  cd $(imgui-bdir); }
imgui-scd(){  cd $(imgui-sdir); }

imgui-cd(){  cd $(imgui-dir)/$1; }

imgui-get(){
   local dir=$(dirname $(imgui-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "imgui" ] && git clone https://github.com/ocornut/imgui.git 

   imgui-fix 
}

imgui-fix(){
   local msg="=== $FUNCNAME :"
   local name=imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp

   [ ! -f "$name" ] && echo $msg from pwd $(pwd) see no $name && return 
   
   perl -pi.orig -e 's,gl3w.h,glew.h,' $name

   diff $name.orig $name
}



imgui-demo(){ vi $(imgui-dir)/imgui.cpp +9757 ; }

imgui-wipe(){
   local bdir=$(imgui-bdir)
   rm -rf $bdir
}

imgui-cmake-ize(){
   cp $(imgui-edir)/CMakeLists.txt $(imgui-sdir)/
}

imgui-cmake(){
   local bdir=$(imgui-bdir)
   mkdir -p $bdir
   imgui-bcd
   cmake $(imgui-sdir) -DCMAKE_INSTALL_PREFIX=$(imgui-idir) -DCMAKE_BUILD_TYPE=Debug 
}

imgui-make(){
    local iwd=$PWD
    imgui-bcd
    make $*
    cd $iwd
}

imgui-install(){
   imgui-make install
}

imgui--(){
   imgui-get

   imgui-wipe
   imgui-cmake-ize
   imgui-cmake
   imgui-make
   [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1
   imgui-install $*
}





########### below funcs used prior to adopting cmake approach above #############

imgui-example(){
   imgui-cd examples/opengl3_example
   imgui-example-make
}

imgui-example-make(){
   [ "$(basename $PWD)" != "opengl3_example" ]  && echo $msg run this from opengl3_example directory && return 
   glfw-
   glfw-export
   make PKG_CONFIG_PATH=$(glfw-prefix)/lib/pkgconfig
}

imgui-srcs(){ cat << EOS
imgui.cpp
imgui.h
imconfig.h
stb_rect_pack.h
stb_textedit.h
stb_truetype.h
examples/opengl3_example/Makefile
examples/opengl3_example/main.cpp
examples/opengl3_example/imgui_impl_glfw_gl3.cpp
examples/opengl3_example/imgui_impl_glfw_gl3.h
examples/libs/gl3w/GL/gl3w.c
examples/libs/gl3w/GL/gl3w.h
examples/libs/gl3w/GL/glcorearb.h
EOS
}

imgui-copy(){
   [ ! -d imgui ] && mkdir imgui
   local srcd=$(imgui-dir)
   local src
   local dst
   for src in $(imgui-srcs) 
   do 
       dst=imgui/$src
       mkdir -p $(dirname $dst)
       cp $(imgui-dir)/$src $dst
   done
}

