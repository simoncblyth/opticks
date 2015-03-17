# === func-gen- : graphics/assimp/assimp fgp graphics/assimp/assimp.bash fgn assimp fgh graphics/assimp
assimp-src(){      echo graphics/assimp/assimp.bash ; }
assimp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(assimp-src)} ; }
assimp-vi(){       vi $(assimp-source) ; }
assimp-env(){      elocal- ; }
assimp-usage(){ cat << EOU

Open Asset Import Library
===========================

Latest version: 3.1.1
Released 2014-06-14 

* http://assimp.sourceforge.net
* http://assimp.sourceforge.net/lib_html/index.html
* http://sourceforge.net/p/assimp/discussion/817654
* http://stackoverflow.com/questions/tagged/assimp?sort=newest

Open Asset Import Library (short name: Assimp) is a portable Open Source
library to import various well-known 3D model formats in a uniform manner.
Written in C++, it is available under a liberal BSD license. There is a C API
as well as bindings to various other languages, including C#/.net, Python and D. 


Development Cycle Adding Extra Material/Surface Handling
---------------------------------------------------------

::

    assimp-cd
    assimp-build

    assimpwrap-
    assimpwrap-test extra


How to add extra material and surface properties ?
------------------------------------------------------

* straightforward to incoporate additional material properties 
  within existing aiMaterial/aiMaterialProperty structs

* not so clear for optical surface properties

  * perhaps make fake materials and put them there ? YES this was approach taken

  * skinsurface reference a single lv
  * bordersurface reference a pair of pv  

  * or could abuse the aiMetadata that is present on every aiNode
    (or just the root node for more global type things)


Yep aiMetadata is meant for simple info, faking materials
looks easiest.



RPATH kludge
--------------

Somehow executables have wrong path to libassimp::

    delta:raytrace blyth$ otool -L AssimpGeometryTest 
    AssimpGeometryTest:
        /usr/local/env/graphics//libassimp.3.dylib (compatibility version 3.0.0, current version 3.1.1)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)

    delta:raytrace blyth$ otool -L RayTrace
    RayTrace:
        /usr/local/env/cuda/OptiX_301/lib/libsutil.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/liboptix.1.dylib (compatibility version 1.0.0, current version 3.0.1)
        /usr/local/env/graphics//libassimp.3.dylib (compatibility version 3.0.0, current version 3.1.1)
        @rpath/libcudart.5.5.dylib (compatibility version 0.0.0, current version 5.5.28)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)


So kludge it with a symbolic link::

    delta:raytrace blyth$ cd /usr/local/env/graphics
    delta:graphics blyth$ ln -s lib/libassimp.3.dylib 


G4DAE extra handling
---------------------

Flat 1 level C++ layout of ColladaLoader and ColladaParser would suggest the
easiest way to add extra element handling for G4DAE material and surface
properties is simply to fork the github assimp.

* http://sourceforge.net/p/assimp/discussion/817653/thread/c3b115cd/


Assimp COLLADA
--------------

Hmm maybe add G4DAELoader that inherits from ColladaLoader
with extra element handling ?

code/ImporterRegistry.cpp::

    127 #ifndef ASSIMP_BUILD_NO_COLLADA_IMPORTER
    128 #   include "ColladaLoader.h"
    129 #endif
    130 #ifndef ASSIMP_BUILD_NO_TERRAGEN_IMPORTER
    131 #   include "TerragenLoader.h"
    132 #endif
    ...
    252 #if (!defined ASSIMP_BUILD_NO_COLLADA_IMPORTER)
    253     out.push_back( new ColladaLoader());
    254 #endif




fork on github
----------------

::

    git clone git://github.com/assimp/assimp.git assimp


config
------

::

    delta:env blyth$ assimp-cmake
    -- The C compiler identification is Clang 6.0.0
    -- The CXX compiler identification is Clang 6.0.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    fatal: Not a git repository (or any of the parent directories): .git
    fatal: Not a git repository (or any of the parent directories): .git
    -- Found PkgConfig: /opt/local/bin/pkg-config (found version "0.28") 
    -- Building a non-boost version of Assimp.
    -- Looking for ZLIB...
    -- checking for module 'zzip-zlib-config'
    --   found zzip-zlib-config, version 1.2.8
    -- Found ZLIB: optimized;/usr/lib/libz.dylib;debug;/usr/lib/libz.dylib
    -- checking for module 'minizip'
    --   package 'minizip' not found
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/graphics/assimp/assimp-3.1.1_build


make
------

::

    Linking CXX executable assimp
    ld: warning: directory not found for option '-L/usr/local/env/graphics/assimp/assimp-3.1.1_build/lib'


install
----------

::

    delta:assimp-3.1.1_build blyth$ assimp-install
    [ 96%] Built target assimp
    [100%] Built target assimp_cmd
    Install the project...
    -- Install configuration: ""
    -- Installing: /usr/local/env/graphics/lib/pkgconfig/assimp.pc
    -- Installing: /usr/local/env/graphics/lib/cmake/assimp-3.1/assimp-config.cmake
    -- Installing: /usr/local/env/graphics/lib/cmake/assimp-3.1/assimp-config-version.cmake
    -- Installing: /usr/local/env/graphics/lib/libassimp.3.1.1.dylib
    -- Installing: /usr/local/env/graphics/lib/libassimp.3.dylib
    -- Installing: /usr/local/env/graphics/lib/libassimp.dylib
    -- Installing: /usr/local/env/graphics/include/assimp/anim.h
    -- Installing: /usr/local/env/graphics/include/assimp/ai_assert.h
    ...
    -- Installing: /usr/local/env/graphics/include/assimp/Compiler/pushpack1.h
    -- Installing: /usr/local/env/graphics/include/assimp/Compiler/poppack1.h
    -- Installing: /usr/local/env/graphics/include/assimp/Compiler/pstdint.h
    -- Installing: /usr/local/env/graphics/bin/assimp
    delta:assimp-3.1.1_build blyth$ 



install paths are broken::

    -- Installing: /usr/local/env/graphics/bin/assimp
    delta:assimp-3.1.1_build blyth$ otool -L /usr/local/env/graphics/bin/assimp
    /usr/local/env/graphics/bin/assimp:
        /usr/local/env/graphics//libassimp.3.dylib (compatibility version 3.0.0, current version 3.1.1)
        /usr/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.2.5)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
    delta:assimp-3.1.1_build blyth$ 



test
-----

::

    delta:assimp_cmd blyth$ ./assimp info $DAE_NAME_DYB
    Launching asset import ...           OK
    Validating postprocessing flags ...  OK
    ERROR: Failed to load file
    assimp info: Unable to load input file /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    delta:assimp_cmd blyth$ 


Have to use the DAE with extra elements removed to succeed to load::

    delta:assimp_cmd blyth$ ./assimp info $DAE_NAME_DYB.noextra.dae 
    Launching asset import ...           OK
    Validating postprocessing flags ...  OK
    Importing file ...                   OK 
       import took approx. 0.51518 seconds

    Memory consumption: 30973491 B
    Nodes:              24461
    Maximum depth       45
    Meshes:             235
    Animations:         0
    Textures (embed.):  0
    Materials:          1
    Cameras:            0
    Lights:             0
    Vertices:           58199
    Faces:              37338
    Bones:              0
    Animation Channels: 0
    Primitive Types:    linestriangles
    Average faces/mesh  158
    Average verts/mesh  247
    Minimum point      (-2400000.000000 -2400000.000000 -2400000.000000)
    Maximum point      (2400000.000000 2400000.000000 2400000.000000)
    Center point       (0.000000 0.000000 0.000000)

    Named Materials:
        'JoinedMaterial_#35'


install test
-------------

::

    delta:~ blyth$ /usr/local/env/graphics/bin/assimp info $DAE_NAME_DYB.noextra.dae 
    dyld: Library not loaded: /usr/local/env/graphics//libassimp.3.dylib
      Referenced from: /usr/local/env/graphics/bin/assimp
      Reason: image not found
    Trace/BPT trap: 5
    delta:~ blyth$ 
    delta:~ blyth$ 
    delta:~ blyth$ otool -L /usr/local/env/graphics/bin/assimp 
    /usr/local/env/graphics/bin/assimp:
        /usr/local/env/graphics//libassimp.3.dylib (compatibility version 3.0.0, current version 3.1.1)
        /usr/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.2.5)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
    delta:~ blyth$ 
    delta:~ blyth$ 







EOU
}

assimp-fork-name(){ echo assimp-fork ; }
assimp-name(){ echo $(assimp-fork-name) ; }
assimp-release-name(){ echo assimp-3.1.1 ; }
assimp-url(){ echo http://downloads.sourceforge.net/project/assimp/assimp-3.1/assimp-3.1.1_no_test_models.zip ; }

assimp-fold(){ echo $(dirname $(assimp-dir)); }
assimp-dir(){ echo $(local-base)/env/graphics/assimp/$(assimp-name) ; }
assimp-prefix(){ echo $(local-base)/env/graphics ; }
assimp-idir(){ echo $(assimp-prefix)/include/assimp ; }
assimp-bdir(){ echo $(assimp-dir)_build ; }
assimp-cd(){  cd $(assimp-dir); }
assimp-bcd(){ cd $(assimp-bdir); }
assimp-icd(){ cd $(assimp-idir); }

assimp-fold-cd(){ cd $(assimp-fold); }

assimp-mate(){ mate $(assimp-dir) ; }
assimp-get(){
   local dir=$(dirname $(assimp-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(assimp-url)
   local zip=$(basename $url)
   local nam=$(assimp-name)
   [ ! -f "$zip" ] && curl -L -O $url 
   [ ! -d "$nam" ] && unzip $zip 
}

assimp-fork-url(){
   case $USER in
     blyth) echo git@github.com:simoncblyth/assimp.git ;;
         *) echo git://github.com/simoncblyth/assimp.git ;;
   esac
} 
assimp-fork-get(){
   local dir=$(dirname $(assimp-dir)) &&  mkdir -p $dir && cd $dir
   local cmd="git clone $(assimp-fork-url) $(assimp-fork-name)"
   echo $cmd
   eval $cmd
}

assimp-rdiff(){
   local rel=${1:-code}
   assimp-fold-cd
   diff -r --brief $(assimp-release-name)/$rel $(assimp-fork-name)/$rel 
}
assimp-diff(){
   local rel=${1:-code/ColladaLoader.cpp}
   assimp-fold-cd
   local cmd="diff $(assimp-release-name)/$rel $(assimp-fork-name)/$rel"
   echo 
   echo $cmd
   eval $cmd
}

assimp-ndiff(){
  local name
  assimp-names | while read name ; do 
     assimp-diff $name 
  done

}


assimp-names(){ cat << EON
code/ColladaExporter.cpp
code/ColladaExporter.h
code/ColladaHelper.h
code/ColladaLoader.cpp
code/ColladaLoader.h
code/ColladaParser.cpp
code/ColladaParser.h
EON
}



assimp-wipe(){
   local bdir=$(assimp-bdir)
   rm -rf $bdir
}


assimp-cmake(){
   local iwd=$PWD
   local bdir=$(assimp-bdir)
   mkdir -p $bdir
   assimp-bcd
   cmake $(assimp-dir) -DCMAKE_INSTALL_PREFIX=$(assimp-prefix)
   cd $iwd
}

assimp-make(){
   local iwd=$PWD
   assimp-bcd
   make $*  && cd $iwd
}

assimp-install(){
   local iwd=$PWD
   assimp-bcd
   #make DESTDIR=$(assimp-prefix) install
   make install  && cd $iwd

   # huh : install says -- Removed runtime path from "/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/env/graphics/bin/assimp"
}


assimp-build(){
   assimp-make
   assimp-install
}


assimp-test(){
   export-
   export-export

   assimp-path-kludge

   local pfx=$(assimp-prefix)
   $pfx/bin/assimp info $DAE_NAME_DYB.noextra.dae 

   # RPATH setup is broken (maybe only for non-default prefix), forcing DLP
}


assimp-path-kludge(){
   local pfx=$(assimp-prefix)
   case $(uname) in 
     Darwin) export DYLD_LIBRARY_PATH=$pfx/lib:$DYLD_LIBRARY_PATH ;;
      Linux) export LD_LIBRARY_PATH=$pfx/lib:$LD_LIBRARY_PATH ;;
   esac
}



assimp-findcmake-(){ cat << EOF

set(Assimp_PREFIX "\$ENV{LOCAL_BASE}/env/graphics")

find_library( Assimp_LIBRARIES 
              NAMES assimp
              PATHS \${Assimp_PREFIX}/lib )

set(Assimp_INCLUDE_DIRS "\${Assimp_PREFIX}/include")
set(Assimp_DEFINITIONS "")

EOF
}


assimp-findcmake(){
  $FUNCNAME- > $ENV_HOME/cmake/Modules/FindAssimp.cmake 
}





