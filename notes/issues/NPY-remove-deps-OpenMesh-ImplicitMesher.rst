NPY-remove-deps-OpenMesh-ImplicitMesher
=========================================

Whilst attempting to remove deps in NPY encounter header finding problem with NGLTF.cpp::

     21 #include <sstream>
     22 
     23 #include "PLOG.hh"
     24 #include "BFile.hh"
     25 
     26 #include "NYGLTF.hpp"
     27 #include "NGLTF.hpp"

NYGLTF.hpp::

     34 
     35 #include "YoctoGL/yocto_gltf.h"
     36 


Reinstated all deps to see the successful commandline by touch and VERBOSE::

    epsilon:npy blyth$ touch NGLTF.cpp ; VERBOSE=1 om 

    [  0%] Building CXX object CMakeFiles/NPY.dir/NGLTF.cpp.o
    /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  
        -DBOOST_ALL_NO_LIB -DBOOST_FILESYSTEM_DYN_LINK -DBOOST_PROGRAM_OPTIONS_DYN_LINK -DBOOST_REGEX_DYN_LINK -DBOOST_SYSTEM_DYN_LINK -DNPY_EXPORTS -DOPTICKS_BRAP -DOPTICKS_DualContouringSample -DOPTICKS_ImplicitMesher -DOPTICKS_NPY -DOPTICKS_OKCONF -DOPTICKS_OpenMesh -DOPTICKS_SYSRAP -DOPTICKS_YoctoGL -DWITH_BOOST_ASIO 
    -I/Users/blyth/opticks/npy 
    -isystem /usr/local/opticks/externals/glm/glm 
    -isystem /usr/local/opticks/include/SysRap 
    -isystem /usr/local/opticks/externals/plog/include 
    -isystem /usr/local/opticks/include/OKConf 
    -isystem /usr/local/opticks/include/BoostRap 
    -isystem /usr/local/opticks_externals/boost/include 
    -isystem /usr/local/opticks/externals/include 
    -isystem /usr/local/opticks/externals/include/YoctoGL 
    -isystem /usr/local/opticks/externals/include/ImplicitMesher 
    -isystem /usr/local/opticks/externals/include/DualContouringSample  
    -fvisibility=hidden 
    -fvisibility-inlines-hidden 
    -fdiagnostics-show-option 
    -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow -g -fPIC   -std=gnu++14 -o CMakeFiles/NPY.dir/NGLTF.cpp.o -c /Users/blyth/opticks/npy/NGLTF.cpp
    [  1%] Linking CXX shared library libNPY.dylib



Seems that are relying on the indiscriminate::

    -isystem /usr/local/opticks/externals/include 

Rather than::

    -isystem /usr/local/opticks/externals/include/YoctoGL 

And somehow skipping the deps prevents that indiscriminate header path being used ?

::

    epsilon:npy blyth$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    openmesh
    plog
    opticksaux
    oimplicitmesher
    odcs
    oyoctogl
    ocsgbsp
    epsilon:npy blyth$ 


oimplicitmesher-cd ; vi CMakeLists.txt::

     01 cmake_minimum_required (VERSION 3.5)
      2 set(name ImplicitMesher)
      3 project(${name} VERSION 0.1.0)
      4 include(OpticksBuildOptions)
      5 
      6 #[=[
      7 Hmm OpticksBuildOptions sets CMAKE_INSTALL_INCLUDEDIR to "include/${name}"
      8 so must override that here rather than from commandline
      9 #]=]
     10 
     11 include(GNUInstallDirs)
     12 set(CMAKE_INSTALL_INCLUDEDIR "externals/include/${name}")
     13 set(CMAKE_INSTALL_LIBDIR     "externals/lib")
     14 set(CMAKE_INSTALL_BINDIR     "lib")
     15 
     ..
     76 
     77 bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
     78 install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


openmesh-cd ; vi CMakeLists.txt

Hmm : actually it is not the CMakeLists.txt of the externals that matters, but rather
how they are found.


cmake/Modules/FindOpenMesh.cmake::

     12 set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals")
     13 
     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include"
     17 )
     18 


Looks like OpenMesh_INCLUDE_DIR is the culprit unspecific.

cd ~/opticks/examples/UseOpenMesh::

   ./go.sh 
   ...
   -- FindOpenMesh.cmake OpenMesh_MODULE     :/Users/blyth/opticks/cmake/Modules/FindOpenMesh.cmake  
   -- FindOpenMesh.cmake OpenMesh_INCLUDE_DIR:/usr/local/opticks/externals/include  
   ...



Rejig to make OpenMesh_INCLUDE_DIR end with OpenMesh::

     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include"
     17 )
     18 

     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include/OpenMesh"
     17 )

Doesnt work as inclusion is expecting the OpenMesh::

    [ 50%] Building CXX object CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o
    /Users/blyth/opticks/examples/UseOpenMesh/UseOpenMesh.cc:22:10: fatal error: 'OpenMesh/Core/IO/MeshIO.hh' file not found
    #include <OpenMesh/Core/IO/MeshIO.hh>
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.
    make[2]: *** [CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o] Error 1
    make[1]: *** [CMakeFiles/UseOpenMesh.dir/all] Error 2
    make: *** [all] Error 2
    [ 50%] Building CXX object CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o
    /Users/blyth/opticks/examples/UseOpenMesh/UseOpenMesh.cc:22:10: fatal error: 'OpenMesh/Core/IO/MeshIO.hh' file not found
    #include <OpenMesh/Core/IO/MeshIO.hh>
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.


Seems its a necessary evil, so prevent its presence being dependent on the current deps and do it always.

npy/CMakeLists.txt::

    455 # some header inclusion expects the package name prefix, eg OpenMesh YoctoGL see notes/issues/NPY-remove-deps-OpenMesh-ImplicitMesher.rst
    456 target_include_directories( ${name} PUBLIC ${OPTICKS_PREFIX}/externals/include )
    45




