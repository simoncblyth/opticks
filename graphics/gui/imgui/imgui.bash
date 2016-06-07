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


FUNCTIONS
-----------



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

imgui-env(){      elocal- ; opticks- ;  }

imgui-edir(){ echo $(opticks-home)/graphics/gui/imgui ; }
imgui-base(){ echo $(opticks-prefix)/externals/imgui ; }

#imgui-prefix(){ echo $(imgui-base)/imgui.install ; }
imgui-prefix(){ echo $(opticks-prefix)/externals ; }

imgui-idir(){ echo $(imgui-prefix) ; }
imgui-bdir(){ echo $(imgui-base)/imgui.build   ; }
imgui-sdir(){ echo $(imgui-base)/imgui ; }
imgui-dir(){  echo $(imgui-base)/imgui ; }

imgui-ecd(){  cd $(imgui-edir); }
imgui-icd(){  cd $(imgui-idir); }
imgui-bcd(){  cd $(imgui-bdir); }
imgui-scd(){  cd $(imgui-sdir); }



imgui-old-base(){ echo $(local-base)/env/graphics/gui ; }
imgui-old-sdir(){ echo $(imgui-old-base)/imgui  ; }
imgui-old-bdir(){ echo $(imgui-old-base)/imgui.build   ; }
imgui-old-scd(){  cd $(imgui-old-sdir); }
imgui-old-bcd(){  cd $(imgui-old-bdir); }

imgui-cd(){  cd $(imgui-dir)/$1; }


imgui-url(){
   #case $USER in
   #  blyth) echo git@github.com:simoncblyth/imgui.git ;;
   #      *) echo git://github.com/simoncblyth/imgui.git ;;
   #esac
   echo git://github.com/simoncblyth/imgui.git
} 

imgui-edit(){ vi $(imgui-edir)/CMakeLists.txt $(opticks-home)/cmake/Modules/FindImGui.cmake ; }

imgui-get(){
   local iwd=$PWD
   local dir=$(dirname $(imgui-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "imgui" ]; then 

       # from my fork : in order to fix the version
       git clone $(imgui-url)

       imgui-fix
   fi 
   cp $(imgui-edir)/CMakeLists.txt $(imgui-sdir)/
   cd $iwd
}

imgui-status(){
  local iwd=$PWD
  imgui-scd
   
  git remote show origin
  git status

  cd $iwd
}


imgui-fix(){
   local msg="=== $FUNCNAME :"
   local name=imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp

   [ ! -f "$name" ] && echo $msg from pwd $(pwd) see no $name && return 
   
   perl -pi.orig -e 's,gl3w.h,glew.h,' $name

   diff $name.orig $name
}

imgui-demo(){ vi $(imgui-dir)/imgui_demo.cpp  ; }

imgui-wipe(){
   local bdir=$(imgui-bdir)
   rm -rf $bdir
}

imgui-cmake(){
  local iwd=$PWD
  local bdir=$(imgui-bdir)
  mkdir -p $bdir
  imgui-bcd

  [ -f CMakeCache.txt ] && echo $msg already configured : imgui-configure to reconfigure  && return 
  cmake -G "$(opticks-cmake-generator)" -DCMAKE_INSTALL_PREFIX=$(imgui-prefix) -DCMAKE_BUILD_TYPE=Debug $(imgui-sdir) 
  cd $iwd
}


imgui-configure()
{
   imgui-wipe
   imgui-cmake $*
}


imgui-make(){
  local iwd=$PWD
  imgui-bcd
  make $*
  cd $iwd
}

imgui--(){
  imgui-get
  imgui-cmake
  imgui-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1
  imgui-make install 
}




