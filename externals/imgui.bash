##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

# === func-gen- : graphics/gui/imgui/imgui fgp externals/imgui.bash fgn imgui fgh graphics/gui/imgui
imgui-src(){      echo externals/imgui.bash ; }
imgui-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(imgui-src)} ; }
imgui-vi(){       vi $(imgui-source) ; }
imgui-usage(){ cat << EOU

ImGUI (MIT) : Immediate Mode GUI
===================================

* https://github.com/ocornut/imgui

* includes an embedded console : developer-centric

These *imgui-* functions add cmake building to ImGui
that allows usage of env/cmake/Modules/FindImGui.cmake 
This is tested by imguitest-


CMake issue reported by Axel 
-------------------------------

The command: imgui-configure gives::

    CMake Error at CMakeLists.txt:10 (include):
      include could not find load file:

        EnvBuildOptions

This is due to dirty OPTICKS_HOME envvar dependency in the CMakeLists.txt::

      1 cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
      2 # this file is copied from imgui-edir to imgui-sdir by imgui-get
      3 set(name ImGui)
      4 project(${name})
      5 
      6 set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
      7                       "$ENV{OPTICKS_HOME}/cmake/Modules"
      8           )
      9 
     10 include(EnvBuildOptions)


Other issues
---------------

Fail to find 

::

    OGLRap.ImGui_INCLUDE_DIRS : /tmp/blyth/opticks20170612/externals/include

    Scanning dependencies of target OGLRap
    [  0%] Building CXX object oglrap/CMakeFiles/OGLRap.dir/StateGUI.cc.o
    /Users/blyth/opticks/oglrap/StateGUI.cc:6:10: fatal error: 'imgui.h' file not found
    #include <imgui.h>
             ^
    1 error generated.



Manual fixup in standard installation, for new layout motivated by greenfield fixes::

    simon:include blyth$ pwd
    /usr/local/opticks/externals/include

    simon:include blyth$ mkdir ImGui
    simon:include blyth$ mv imconfig.h imgui.h imgui_impl_glfw_gl3.h ImGui/
    simon:include blyth$ 


::

    delta:oglrap blyth$ l /tmp/blyth/opticks20170612/externals/imgui/imgui/
    total 1992
    -rw-r--r--   1 blyth  wheel    3456 Jun 12 15:15 CMakeLists.txt
    -rw-r--r--   1 blyth  wheel    1106 Jun 12 15:15 LICENSE
    -rw-r--r--   1 blyth  wheel   15735 Jun 12 15:15 README.md
    drwxr-xr-x  17 blyth  wheel     578 Jun 12 15:15 examples
    drwxr-xr-x   9 blyth  wheel     306 Jun 12 15:15 extra_fonts
    -rw-r--r--   1 blyth  wheel    2240 Jun 12 15:15 imconfig.h
    -rw-r--r--   1 blyth  wheel  406005 Jun 12 15:15 imgui.cpp
    -rw-r--r--   1 blyth  wheel  113489 Jun 12 15:15 imgui.h
    -rw-r--r--   1 blyth  wheel  109946 Jun 12 15:15 imgui_demo.cpp
    -rw-r--r--   1 blyth  wheel  105674 Jun 12 15:15 imgui_draw.cpp
    -rw-r--r--   1 blyth  wheel   45273 Jun 12 15:15 imgui_internal.h
    -rw-r--r--   1 blyth  wheel   17059 Jun 12 15:15 stb_rect_pack.h
    -rw-r--r--   1 blyth  wheel   48747 Jun 12 15:15 stb_textedit.h
    -rw-r--r--   1 blyth  wheel  127748 Jun 12 15:15 stb_truetype.h
    delta:oglrap blyth$ 




Greenfield fail
------------------

::

    === -opticks-installer : imgui
    Cloning into 'imgui'...
    remote: Counting objects: 9001, done.
    remote: Total 9001 (delta 0), reused 0 (delta 0), pack-reused 9000
    Receiving objects: 100% (9001/9001), 10.14 MiB | 147.00 KiB/s, done.
    Resolving deltas: 100% (6225/6225), done.
    Checking connectivity... done.
    13c13
    < #include <GL/gl3w.h>
    ---
    > #include <GL/glew.h>
    -- The C compiler identification is AppleClang 6.0.0.6000057
    -- The CXX compiler identification is AppleClang 6.0.0.6000057
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring ImGui
    APPLE
     DEFINITIONS :  
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks20170612/externals/imgui/imgui.build
    Scanning dependencies of target ImGui
    [ 20%] Building CXX object CMakeFiles/ImGui.dir/imgui.cpp.o
    [ 40%] Building CXX object CMakeFiles/ImGui.dir/imgui_draw.cpp.o
    [ 60%] Building CXX object CMakeFiles/ImGui.dir/imgui_demo.cpp.o
    [ 80%] Building CXX object CMakeFiles/ImGui.dir/examples/opengl3_example/imgui_impl_glfw_gl3.cpp.o
    /tmp/blyth/opticks20170612/externals/imgui/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp:13:10: fatal error: 'GL/glew.h' file not found
#include <GL/glew.h>
             ^
    1 error generated.
    make[2]: *** [CMakeFiles/ImGui.dir/examples/opengl3_example/imgui_impl_glfw_gl3.cpp.o] Error 1
    make[1]: *** [CMakeFiles/ImGui.dir/all] Error 2
    make: *** [all] Error 2



Windows VS2015
------------------


Finding GLFW and GLEW
~~~~~~~~~~~~~~~~~~~~~~~~~


Rejigged glfw- and glew- to install into standard locations on windows.


link.exe missing OpenGL symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    imgui_impl_glfw_gl3.obj : error LNK2019: unresolved external symbol __imp__glBindTexture@8 referenced in function "bool __cdecl ImGui_ImplGlfwGL                                                         3_CreateDeviceObjects(void)" (?ImGui_ImplGlfwGL3_CreateDeviceObjects@@YA_NXZ) [C:\usr\local\opticks\externals\imgui\imgui.build\ImGui.vcxproj]
    imgui_impl_glfw_gl3.obj : error LNK2019: unresolved external symbol __imp__glBlendFunc@8 referenced in function "void __cdecl ImGui_ImplGlfwGL3_                                                         RenderDrawLists(struct ImDrawData *)" (?ImGui_ImplGlfwGL3_RenderDrawLists@@YAXPAUImDrawData@@@Z) [C:\usr\local\opticks\externals\imgui\imgui.bui                                                         ld\ImGui.vcxproj]
     



need export header and linker library .lib for this
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


*  http://gernotklingler.com/blog/creating-using-shared-libraries-different-compilers-different-operating-systems/
* https://github.com/Kitware/CMake/blob/master/Modules/GenerateExportHeader.cmake




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

imgui-env(){      olocal- ; opticks- ;  }

imgui-edir(){ echo $(opticks-home)/externals/imgui ; }
imgui-base(){ echo $(opticks-prefix)/externals/imgui ; }
imgui-prefix(){ echo $(opticks-prefix) ; }

imgui-idir(){ echo $(imgui-prefix) ; }
imgui-bdir(){ echo $(imgui-base)/imgui.build   ; }
imgui-sdir(){ echo $(imgui-base)/imgui ; }
imgui-dir(){  echo $(imgui-base)/imgui ; }

imgui-ecd(){  cd $(imgui-edir); }
imgui-icd(){  cd $(imgui-idir); }
imgui-bcd(){  cd $(imgui-bdir); }
imgui-scd(){  cd $(imgui-sdir); }

imgui-EnvBuildOptions(){ echo $OPTICKS_HOME/cmake/Modules/EnvBuildOptions.cmake ; }
imgui-EnvBuildOptions-ls(){ ls -l $(${FUNCNAME/-ls}) ; }

imgui-info(){ cat << EOI

$FUNCNAME
================

    imgui-prefix : $(imgui-prefix)
    imgui-edir : $(imgui-edir)
    imgui-idir : $(imgui-idir)
    imgui-bdir : $(imgui-bdir)
    imgui-sdir : $(imgui-sdir)
    imgui-dir  : $(imgui-dir)

    OPTICKS_HOME : $OPTICKS_HOME
    opticks-home  : $(opticks-home)

    imgui-EnvBuildOptions : $(imgui-EnvBuildOptions)
    imgui-EnvBuildOptions-ls : $(imgui-EnvBuildOptions-ls)

   
    imgui-url  : $(imgui-url)

EOI
}


imgui-cd(){  cd $(imgui-dir)/$1; }


imgui-url(){
   #case $USER in
   #  blyth) echo git@github.com:simoncblyth/imgui.git ;;
   #      *) echo git://github.com/simoncblyth/imgui.git ;;
   #esac
   #echo git://github.com/simoncblyth/imgui.git
   echo http://github.com/simoncblyth/imgui.git
} 

imgui-edit(){ vi $(imgui-edir)/CMakeLists.txt $(opticks-home)/cmake/Modules/FindImGui.cmake ; }

imgui-get(){
   local iwd=$PWD
   local dir=$(dirname $(imgui-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "imgui" ]; then 

       # from my fork : in order to pin the version
       git clone $(imgui-url)

       #imgui-fix
   fi 
  
   ## cp $(imgui-edir)/CMakeLists.txt $(imgui-sdir)/
   ## dont do this, just add it to the forked imgui 


   cd $iwd
}

imgui-cmake-diff(){
   local cmd="diff $(imgui-edir)/CMakeLists.txt $(imgui-sdir)/CMakeLists.txt"
   echo $cmd
   eval $cmd
}
imgui-cmake-copyback(){
   cp $(imgui-sdir)/CMakeLists.txt $(imgui-edir)/CMakeLists.txt 
}
imgui-cmake-copyin(){
   cp $(imgui-edir)/CMakeLists.txt $(imgui-sdir)/CMakeLists.txt 
}



imgui-status(){
  local iwd=$PWD
  imgui-scd
   
  git remote show origin
  git status

  cd $iwd
}

#  just make the change in the fork
#imgui-fix(){
#   local msg="=== $FUNCNAME :"
#   local name=imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp
#
#   [ ! -f "$name" ] && echo $msg from pwd $(pwd) see no $name && return 
#   
#   perl -pi.orig -e 's,gl3w.h,glew.h,' $name
#
#   diff $name.orig $name
#}

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
  cmake \
      -G "$(opticks-cmake-generator)" \
      -DOPTICKS_PREFIX=$(opticks-prefix) \
      -DCMAKE_INSTALL_PREFIX=$(imgui-prefix) \
      -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
      -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
      -DCMAKE_BUILD_TYPE=Debug \
      $(imgui-sdir) 
  cd $iwd
}


imgui-configure()
{
   imgui-wipe
   imgui-cmake $*
}


imgui-config(){ echo Debug ; }
imgui-make(){
  local iwd=$PWD
  imgui-bcd

  #make $*
  cmake --build . --config $(imgui-config) --target ${1:-install}


  cd $iwd
}

imgui--(){

  local iwd=$PWD

  imgui-info 

  imgui-get
  imgui-cmake
  imgui-make install 
   
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1

  imgui-pc 


  cd $iwd
}

imgui-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/imgui.pc ; }
imgui-pc-(){ 

   oc-
   oc-variables-
   cat << EOP

Name: ImGui
Description: OpenGL Graphics Interface 
Version: 0.1.0

Cflags:  -I\${includedir}
Libs: -L\${libdir} -lImGui -lGLEW
Requires: 

EOP
}

imgui-pc () 
{ 
    local msg="=== $FUNCNAME :";
    local path=$(imgui-pc-path);
    local dir=$(dirname $path);
    [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir;
    echo $msg path $path 
    imgui-pc- > $path
}





