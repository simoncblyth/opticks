OPTICKS_CONFIG_Client_install_misses_FindOpticks_cmake
========================================================


HMM, not just Client the "cmake" project was missing from om-subs for both 
Debug and Client. Plus it needed some updates for FindOpticks.cmake::

    (ok) A[blyth@localhost cmake]$ om
    === om-make-one : cmake           /home/blyth/opticks/cmake                                    /data1/blyth/local/opticks_Debug/build/cmake                 
    [ 50%] Building CXX object tests/CMakeFiles/OpticksCMakeModulesTest.dir/OpticksCMakeModulesTest.cc.o
    [100%] Linking CXX executable OpticksCMakeModulesTest
    [100%] Built target OpticksCMakeModulesTest
    [100%] Built target OpticksCMakeModulesTest
    Install the project...
    -- Install configuration: "Debug"
    -- Up-to-date: /data1/blyth/local/opticks_Debug/cmake/Modules/EchoTarget.cmake
    -- Up-to-date: /data1/blyth/local/opticks_Debug/cmake/Modules/TopMetaTarget.cmake
    -- Up-to-date: /data1/blyth/local/opticks_Debug/cmake/Modules/FindG4.cmake
    -- Up-to-date: /data1/blyth/local/opticks_Debug/cmake/Modules/FindGLM.cmake
    -- Up-to-date: /data1/blyth/local/opticks_Debug/cmake/Modules/FindImGui.cmake
    CMake Error at cmake_install.cmake:46 (file):
      file INSTALL cannot find
      "/home/blyth/opticks/cmake/Modules/FindOpenMesh.cmake": No such file or
      directory.


    make: *** [Makefile:110: install] Error 1
    === om-one-or-all make : non-zero rc 2
    (ok) A[blyth@localhost cmake]$ 

