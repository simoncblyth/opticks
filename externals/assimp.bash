# === func-gen- : graphics/assimp/assimp fgp externals/assimp.bash fgn assimp fgh graphics/assimp
assimp-src(){      echo externals/assimp.bash ; }
assimp-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(assimp-src)} ; }
assimp-vi(){       vi $(assimp-source) ; }
assimp-usage(){ cat << EOU

Open Asset Import Library
===========================

Version: 3.1.1
Released 2014-06-14 

* http://assimp.sourceforge.net
* http://assimp.sourceforge.net/lib_html/index.html
* http://sourceforge.net/p/assimp/discussion/817654
* http://stackoverflow.com/questions/tagged/assimp?sort=newest

Open Asset Import Library (short name: Assimp) is a portable Open Source
library to import various well-known 3D model formats in a uniform manner.
Written in C++, it is available under a liberal BSD license. There is a C API
as well as bindings to various other languages, including C#/.net, Python and D. 


TODO : tone down loading verbosity
------------------------------------

Currently dumping many thousands of lines...

::

    delta:opticks blyth$ op --dsst -G
    288 -rwxr-xr-x  1 blyth  staff  143804 Jul  3 11:51 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --dsst -G
    2017-07-03 11:53:06.260 INFO  [2788361] [AssimpGGeo::load@131] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:4448:4456 ctrl volnames verbosity 0
    2017-07-03 11:53:06.260 INFO  [2788361] [AssimpImporter::import@195] AssimpImporter::import path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [REFLECTIVITY0xccef2e8] [REFLECTIVITY] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [RINDEX0xc0d2610] [RINDEX] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [3] [g4dae_opticalsurface_finish] 
    ColladaLoader::BuildMaterialsExtras AddProperty [3] [g4dae_opticalsurface_finish] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [1] [g4dae_opticalsurface_model] 
    ColladaLoader::BuildMaterialsExtras AddProperty [1] [g4dae_opticalsurface_model] 
    ...
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [ABSLENGTH0xc358ff8] [ABSLENGTH] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [ABSLENGTH0xc34ea38] [ABSLENGTH] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [ABSLENGTH0xc0ff7e8] [ABSLENGTH] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [RINDEX0xbf9fc40] [RINDEX] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [ABSLENGTH0xc1782e8] [ABSLENGTH] 
    ColladaLoader::BuildMaterialsExtras (all) AddProperty [RINDEX0xc356448] [RINDEX] 
    2017-07-03 11:53:06.906 INFO  [2788361] [AssimpImporter::Summary@112] AssimpImporter::import DONE
    2017-07-03 11:53:06.906 INFO  [2788361] [AssimpImporter::Summary@113] AssimpImporter::info m_aiscene  NumMaterials 78 NumMeshes 249





Assimp on Windows
-------------------

* http://www.assimp.org/lib_html/install.html


Mesh Processing Flags
-----------------------

* http://assimp.sourceforge.net/lib_html/postprocess_8h.html

::

    aiProcess_JoinIdenticalVertices  : TODO: check if this is per mesh 
    aiProcess_Triangulate            : triangulate any quads


Development Cycle Adding Extra Material/Surface Handling
---------------------------------------------------------

::

    assimp-cd
    assimp-build

    assimprap-
    assimprap-test extra


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



CMake on Windows
---------------------

Windows install puts dll in bin::

    -- Installing: C:/msys64/usr/local/opticks/externals/lib/libassimp.dll.a
    -- Installing: C:/msys64/usr/local/opticks/externals/bin/libassimp.dll
    -- Installing: C:/msys64/usr/local/opticks/externals/include/assimp/anim.h




Warning 
--------

::

   CMake Warning (dev):
   Policy CMP0042 is not set: MACOSX_RPATH is enabled by default.  Run "cmake
   --help-policy CMP0042" for policy details.  Use the cmake_policy command to
   set the policy and suppress this warning.

   MACOSX_RPATH is not specified for the following targets:

   assimp



Warnings
---------

Boost not found by assimp cmake::

    -- Detecting CXX compile features - done
    -- Found PkgConfig: /opt/local/bin/pkg-config (found version "0.28") 
    -- Building a non-boost version of Assimp.

Linker expecting lib dir in build directory?::

    [ 99%] Building CXX object tools/assimp_cmd/CMakeFiles/assimp_cmd.dir/Export.cpp.o
    [100%] Linking CXX executable assimp
    ld: warning: directory not found for option '-L/usr/local/opticks/externals/assimp/assimp-fork.build/lib'
    [100%] Built target assimp_cmd


Redoing make install gives other RPATH related errors::

    -- Up-to-date: /usr/local/opticks/externals/assimp/assimp/bin/assimp
    error: /opt/local/bin/install_name_tool: 
         no LC_RPATH load command with path: /usr/local/opticks/externals/assimp/assimp-fork.build 
         found in: /usr/local/opticks/externals/assimp/assimp/bin/assimp (for architecture x86_64), 
         required for specified option "-delete_rpath /usr/local/opticks/externals/assimp/assimp-fork.build"
    error: /opt/local/bin/install_name_tool: 
         no LC_RPATH load command with path: /usr/local/opticks/externals/assimp/assimp-fork.build/lib 
         found in: /usr/local/opticks/externals/assimp/assimp/bin/assimp (for architecture x86_64), 
         required for specified option "-delete_rpath /usr/local/opticks/externals/assimp/assimp-fork.build/lib"
    error: /opt/local/bin/install_name_tool: 
         no LC_RPATH load command with path: /usr/local/opticks/externals/assimp/assimp-fork.build/code 
         found in: /usr/local/opticks/externals/assimp/assimp/bin/assimp (for architecture x86_64), 
         required for specified option "-delete_rpath /usr/local/opticks/externals/assimp/assimp-fork.build/code"



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


fork on github
----------------

::

    git clone git://github.com/assimp/assimp.git assimp


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

assimp-env(){      olocal- ; opticks- ;  }
assimp-fork-name(){ echo assimp-fork ; }
assimp-name(){ echo $(assimp-fork-name) ; }

assimp-release-name(){ echo assimp-3.1.1 ; }
assimp-release-url(){  echo http://downloads.sourceforge.net/project/assimp/assimp-3.1/assimp-3.1.1_no_test_models.zip ; }
assimp-dev-url(){      echo git@github.com:simoncblyth/assimp.git ; } 
assimp-url(){          echo http://github.com/simoncblyth/assimp.git ; } 

assimp-doc(){ open   ; }


assimp-fold(){ echo $(dirname $(assimp-dir)); }

assimp-base(){   echo $(opticks-prefix)/externals/assimp ; }
assimp-dir(){    echo $(assimp-base)/$(assimp-name) ; }
#assimp-prefix(){ echo $(assimp-base)/assimp ; }
assimp-prefix(){ echo $(opticks-prefix)/externals ; }
assimp-edit(){ vi $(opticks-home)/cmake/Modules/FindAssimp.cmake ; }

assimp-idir(){ echo $(assimp-prefix)/include/assimp ; }
assimp-bdir(){ echo $(assimp-dir).build ; }

assimp-cd(){  cd $(assimp-dir); }
assimp-bcd(){ cd $(assimp-bdir); }
assimp-icd(){ cd $(assimp-idir); }

assimp-fold-cd(){ cd $(assimp-fold); }

assimp-release-get(){
   local iwd=$PWD
   local dir=$(dirname $(assimp-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(assimp-release-url)
   local zip=$(basename $url)
   local nam=$(assimp-release-name)
   [ ! -f "$zip" ] && curl -L -O $url 
   [ ! -d "$nam" ] && unzip $zip 
   cd $iwd
}

assimp-get(){
   local iwd=$PWD
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(assimp-dir)) &&  mkdir -p $dir && cd $dir
   local dst=$(assimp-fork-name)
   local cmd="git clone $(assimp-url) $dst"
  
   if [ -d "$dst" ]; then
       echo $msg already did \"$cmd\" from $PWD
   else
       echo $cmd
       eval $cmd
   fi
   cd $iwd
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

assimp-configure(){
   assimp-wipe
   assimp-cmake $* 
}


assimp-cmake(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local bdir=$(assimp-bdir)
   mkdir -p $bdir
   [ -f $bdir/CMakeCache.txt ] && echo $msg configured already : use assimp-configure to reconfigure  && return 
   assimp-bcd

   local opts=""
   [ "$(uname)" == "Darwin" ] && opts="-DCMAKE_MACOSX_RPATH:BOOL=ON" 

   cmake \
        -G "$(opticks-cmake-generator)" \
        -DCMAKE_INSTALL_PREFIX=$(assimp-prefix)  \
        -DASSIMP_BUILD_TESTS=OFF  \
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF  \
         $opts \
         $(assimp-dir) 

   cd $iwd
}



assimp-config(){ echo "Debug" ; }

assimp-make(){
   local iwd=$PWD
   assimp-bcd

   #make $*  
   cmake --build . --config $(assimp-config) --target ${1:-install}

   cd $iwd
}

assimp--() {
   assimp-get
   assimp-cmake
   assimp-make
   assimp-make install

   assimp-rpath-kludge
}

assimp-build(){
   assimp-make
   assimp-make install
}

assimp-libname(){
   case $(uname -s) in 
     Darwin) echo libassimp.3.dylib ;;
      Linux)  echo libassimp.so.3 ;;
          *)  echo libassimp.dll.3 ;;
   esac
}

assimp-rpath-kludge()
{
   local msg="=== $FUNCNAME :"
   local iwd=$PWD

   cd $(assimp-prefix)

   local lib=$(assimp-libname)

   if [ -x "$lib" ]; then
       echo $msg already present : $lib
   else
      if [ -f "lib/$lib" ]; then 
          echo $msg proceed with symbolicating lib/$lib
          ln -s lib/$lib
      else
          echo $msg cannot proceed as target lib/$lib is missing 
      fi
   fi

   cd $iwd
}


assimp-test(){
   export-
   export-export

   local pfx=$(assimp-prefix)
   $pfx/bin/assimp info $DAE_NAME_DYB.noextra.dae 

}

