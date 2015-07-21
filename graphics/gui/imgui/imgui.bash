# === func-gen- : graphics/gui/imgui/imgui fgp graphics/gui/imgui/imgui.bash fgn imgui fgh graphics/gui/imgui
imgui-src(){      echo graphics/gui/imgui/imgui.bash ; }
imgui-source(){   echo ${BASH_SOURCE:-$(env-home)/$(imgui-src)} ; }
imgui-vi(){       vi $(imgui-source) ; }
imgui-env(){      elocal- ; }
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


imgui-edir(){ echo $(env-home)/graphics/gui/imgui ; }
imgui-idir(){ echo $(local-base)/env/graphics/gui/imgui.install ; }
imgui-bdir(){ echo $(local-base)/env/graphics/gui/imgui.build   ; }
imgui-sdir(){ echo $(local-base)/env/graphics/gui/imgui ; }
imgui-dir(){  echo $(local-base)/env/graphics/gui/imgui ; }

imgui-ecd(){  cd $(imgui-edir); }
imgui-icd(){  cd $(imgui-idir); }
imgui-bcd(){  cd $(imgui-bdir); }
imgui-scd(){  cd $(imgui-sdir); }

imgui-cd(){  cd $(imgui-dir)/$1; }

imgui-get(){
   local dir=$(dirname $(imgui-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "imgui" ] && git clone https://github.com/ocornut/imgui.git 
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

