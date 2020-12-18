CMake_dependency_include_dirs_issue_reported_by_Hans
======================================================

Hans reports having to add::

   include_directories($ENV{OPTICKS_HOME}/npy)

To::

   optickscore/CMakeLists.txt
   opticksgeo/CMakeLists.txt

Whereas I see no such need.


Resolved : by deleting lines including former GLTF related headers
--------------------------------------------------------------------

There was no CMake issue, just the inclusion of some stale headers.


Make is easier to reproduce such problems in future with : om-prefix-clean
-----------------------------------------------------------------------------

om.bash::

     177 om-prefix-clean-dirs-(){ cat << EOD
     178 lib
     179 lib64
     180 build
     181 include
     182 EOD
     183 }
     184 om-prefix-clean-notes(){ cat << EON
     185 
     186 See notes/issues/CMake_dependency_include_dirs_issue_reported_by_Hans.rst 
     187 
     188 If users report CMake dependency issues, the cause might not be due to CMake 
     189 but rather due to failures to include stale headers.  In order to find 
     190 these bad inclusions a deeper clean than om-cleaninstall is needed.
     191 In that case use::
     192 
     193     om-prefix-clean 
     194 
     195 EON
     196 }
     197 om-prefix-clean(){
     198    local msg="=== $FUNCNAME :"
     199    cd $(om-prefix)
     200    pwd
     201    local dirs=$(om-prefix-clean-dirs-)
     202    echo $msg om-prefix : $(om-prefix)  
     203    echo $msg pwd       : $(pwd)  
     204    echo $msg om-prefix-clean-dirs- : $dirs  
     205 
     206    local ans
     207    read -p "$msg enter YES to proceed with deletion of prefix dirs : " ans
     208 
     209    if [ "$ans" == "YES" ]; then
     210        local cmd
     211        for dir in $dirs ; do
     212            cmd="rm -rf $dir"
     213            echo $cmd
     214            eval $cmd
     215        done
     216    else
     217        echo $msg SKIPPED 
     218    fi
     219 }





Attempt to reproduce the issue
-------------------------------

1. try a clean install::

   cd ~/opticks
   om-
   om-cleaninstall

The cleaninstall deletes the build dirs before doing the builds so 
it should avoid problems from old generated .cmake files 
from the prior tree of projects. 

BUT, this succeeds to build with no problems.


2. try a deeper clean::

    [blyth@localhost opticks]$ opticks-cd ; echo $(opticks-prefix) ; pwd 
    /home/blyth/local/opticks
    /home/blyth/local/opticks
    [blyth@localhost opticks]$ mv lib lib.old
    [blyth@localhost opticks]$ mv bin bin.old
    [blyth@localhost opticks]$ mv build build.old
    [blyth@localhost opticks]$ mv lib64 lib64.old
    [blyth@localhost opticks]$ mv include include.old


    [blyth@localhost opticks]$ cd ~/opticks ; om- ; om-cleaninstall


Oops building misses setup::

    === om-env : normal running
    -bash: /home/blyth/local/opticks/bin/opticks-setup.sh: No such file or directory

    [blyth@localhost opticks]$ mkdir bin
    [blyth@localhost opticks]$  cp bin.old/opticks-setup.sh bin/


Now get somewhere. okc fails to include former header NGLTF.hpp (which is no longer needed following removal of YoctoGL)::

    [ 32%] Building CXX object CMakeFiles/OpticksCore.dir/Bookmarks.cc.o
    [ 33%] Building CXX object CMakeFiles/OpticksCore.dir/CompositionCfg.cc.o
    [ 34%] Building CXX object CMakeFiles/OpticksCore.dir/Composition.cc.o
    /home/blyth/opticks/optickscore/Opticks.cc:69:21: fatal error: NGLTF.hpp: No such file or directory
     #include "NGLTF.hpp"
                         ^
    compilation terminated.

Delete that line and proceed with cleaninstall::

    om
    om-cleaninstall +    ## + means next sub projects in teh sequence


okg failing to include former header HitsNPY.hpp (no longer used as that was for old sensor and resource handling)::

    [ 30%] Building CXX object CMakeFiles/OpticksGeo.dir/OpticksGun.cc.o
    [ 38%] Building CXX object CMakeFiles/OpticksGeo.dir/OpticksGen.cc.o
    [ 46%] Building CXX object CMakeFiles/OpticksGeo.dir/OpticksIdx.cc.o
    /home/blyth/opticks/opticksgeo/OpticksIdx.cc:30:23: fatal error: HitsNPY.hpp: No such file or directory
     #include "HitsNPY.hpp"
                           ^
    compilation terminated.
    make[2]: *** [CMakeFiles/OpticksGeo.dir/OpticksIdx.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs...


Again, delete that line and proceed with cleaninstall::

    om
    om-cleaninstall +    ## + means next sub projects in teh sequence


Get to okg4::

    === om-make-one : okg4            /home/blyth/opticks/okg4                                     /home/blyth/local/opticks/build/okg4                         
    Scanning dependencies of target OKG4
    [ 85%] Built target OKG4Test
    ...
    /home/blyth/opticks/okg4/tests/OKX4Test.cc:38:23: fatal error: GGeoGLTF.hh: No such file or directory
     #include "GGeoGLTF.hh"
                           ^

Yet again, delete that line and proceed with cleaninstall::

    om
    om-cleaninstall +    ## + means next sub projects in teh sequence

::

    === om-make-one : g4ok            /home/blyth/opticks/g4ok                                     /home/blyth/local/opticks/build/g4ok                         
    ...
    /home/blyth/opticks/g4ok/G4Opticks.cc:59:23: fatal error: GGeoGLTF.hh: No such file or directory
     #include "GGeoGLTF.hh"
                           ^

Same again, delete the line and proceed::

    om
    om-cleaninstall + 


And the install completes.



Tidy up::

    opticks-cd
    [blyth@localhost opticks]$ rm -rf lib64.old
    [blyth@localhost opticks]$ rm -rf include.old
    [blyth@localhost opticks]$ rm -rf lib.old
    [blyth@localhost opticks]$ rm -rf build.old


