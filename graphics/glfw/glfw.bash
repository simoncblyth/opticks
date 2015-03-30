# === func-gen- : graphics/glfw/glfw fgp graphics/glfw/glfw.bash fgn glfw fgh graphics/glfw
glfw-src(){      echo graphics/glfw/glfw.bash ; }
glfw-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfw-src)} ; }
glfw-vi(){       vi $(glfw-source) ; }
glfw-env(){      elocal- ; }
glfw-usage(){ cat << EOU

GLFW
======

* http://www.glfw.org

GLFW is an Open Source, multi-platform library for creating windows with OpenGL
contexts and receiving input and events. It is easy to integrate into existing
applications and does not lay claim to the main loop.

Version 3.1.1 released on March 19, 2015

pkg-config
------------

Uppercased the glfw3.pc in attempt to get oglplus- cmake to find it, 
to no avail. 

::

    delta:oglplustest blyth$ ll $(glfw-idir)/lib/pkgconfig/
    total 8
    -rw-r--r--  1 blyth  staff  422 Mar 24 13:00 glfw3.pc
    drwxr-xr-x  5 blyth  staff  170 Mar 24 13:00 ..

    After renaming pkg-config from commamdline can find it, but not cmake 

    delta:~ blyth$ PKG_CONFIG_PATH=$(glfw-idir)/lib/pkgconfig pkg-config GLFW3 --modversion
    3.1.1

    delta:pkgconfig blyth$ glfw-;glfw-pc GLFW3 --libs --static
    -L/usr/local/env/graphics/glfw/3.1.1/lib -lglfw3 -framework Cocoa -framework OpenGL -framework IOKit -framework CoreFoundation -framework CoreVideo 

    delta:pkgconfig blyth$ glfw-;glfw-pc GLFW3 --libs 
    -L/usr/local/env/graphics/glfw/3.1.1/lib -lglfw3 


input events
-------------

::

    glfw-bcd
    cd tests
    ./events   # blank window demo of all events 

    delta:tests blyth$ pwd
    /usr/local/env/graphics/glfw/glfw-3.1.1.build/tests
    delta:tests blyth$ ./events -h
    Library initialized
    Usage: events [-f] [-h] [-n WINDOWS]
    Options:
      -f use full screen
      -h show this help
      -n the number of windows to create

    00000433 to 2 at 70.437: Cursor position: 302.957031 217.011719
    00000434 to 2 at 70.454: Cursor position: 310.250000 212.843750
    00000435 to 2 at 70.471: Cursor position: 312.332031 211.800781

Observations

* origin of cursor coordinates is the top left of current focused window,
* goes negative to the left and up, increases to right and down 
  and continues beyond window dimensions 



trackball
------------

*  https://github.com/ishikawash/glfw-example/blob/master/normal_map_demo/trackball.hpp



external events, eg messages from UDP,  ZeroMQ
------------------------------------------------

* https://github.com/elmindreda/Wendy


GLEQ : simple single header event queue for GLFW 3
----------------------------------------------------------

By the author of GLFW

* https://github.com/elmindreda/gleq
* https://github.com/elmindreda/gleq/blob/master/test.c 
* https://github.com/elmindreda/gleq/blob/master/test.c 

When a window is tracked by GLEQ it handles setting all the 
callbacks, which populate a unionized event struct. This
yields a clean interface for consuming input events which 
keeps the callbacks out of sight.






EOU
}
glfw-sdir(){ echo $(env-home)/graphics/glfw ; }
glfw-dir(){  echo $(local-base)/env/graphics/glfw/$(glfw-name) ; }
glfw-bdir(){ echo $(glfw-dir).build ; }
glfw-idir(){ echo $(local-base)/env/graphics/glfw/$(glfw-version) ; }

glfw-scd(){ cd $(glfw-sdir); }
glfw-cd(){  cd $(glfw-dir); }
glfw-bcd(){ cd $(glfw-bdir); }
glfw-icd(){ cd $(glfw-idir); }

glfw-pc(){
  PKG_CONFIG_PATH=$(glfw-idir)/lib/pkgconfig pkg-config GLFW3 $*
}

glfw-pc-kludge(){
   cd $(glfw-idir)/lib/pkgconfig
   mv glfw3.pc GLFW3.pc
}

glfw-version(){ echo 3.1.1 ; }
glfw-name(){ echo glfw-$(glfw-version) ; }
glfw-url(){  echo http://downloads.sourceforge.net/project/glfw/glfw/$(glfw-version)/$(glfw-name).zip ; }

glfw-get(){
   local dir=$(dirname $(glfw-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(glfw-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip 

}


glfw-wipe(){
  local bdir=$(glfw-bdir)
  rm -rf $bdir
}

glfw-cmake(){
  local iwd=$PWD

  local bdir=$(glfw-bdir)
  mkdir -p $bdir

  glfw-bcd
  cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$(glfw-idir) $(glfw-dir)

  cd $iwd
}

glfw-make(){
  local iwd=$PWD

  glfw-bcd
  make $* 

  cd $iwd
}


