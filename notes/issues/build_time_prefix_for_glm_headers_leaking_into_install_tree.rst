build_time_prefix_for_glm_headers_leaking_into_install_tree
============================================================


Context
---------

* :doc:`../docker_junosw_opticks_container_build_shakedown`

Symptom
--------

j+o build inside container fails to config::

      Imported target "Opticks::G4CX" includes non-existent path

        "/data1/blyth/local/opticks_Debug/externals/glm/glm"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.


Fix
----

::

    A[blyth@localhost opticks]$ git diff sysrap/CMakeLists.txt
    diff --git a/sysrap/CMakeLists.txt b/sysrap/CMakeLists.txt
    index 4de44a5f7..92585e7a1 100644
    --- a/sysrap/CMakeLists.txt
    +++ b/sysrap/CMakeLists.txt
    @@ -52,11 +52,6 @@ find_package(Custom4 CONFIG)
     message(STATUS "${name} Custom4_FOUND:${Custom4_FOUND}  " )
     
     
    -include_directories(
    -    ${OPTICKS_PREFIX}/externals/glm/glm
    -)
    -
    -
     set(WITH_SLOG YES)
     
     
    @@ -697,10 +692,13 @@ endif()
     target_include_directories( ${name} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        ${CUDA_INCLUDE_DIRS}
    -   ${OPTICKS_PREFIX}/externals/glm/glm
    +   $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
    +   $<INSTALL_INTERFACE:externals/glm/glm>
     )
     

Fix is to distingish the include directory during build and after install using::

      $<BUILD_INTERFACE:>
      $<INSTALL_INTERFACE:>


Any more ?
------------

::

    A[blyth@localhost opticks]$ find . -name CMakeLists.txt -exec grep -H OPTICKS_PREFIX {} \;
    ./CSG/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./CSG/CMakeLists.txt:    ${OPTICKS_PREFIX}/include/SysRap
    ./CSGOptiX/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./sysrap/CMakeLists.txt:   $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>



     ## exclude dead code and examples
    ./qudarap/CMakeLists.txt:    epsilon:qudarap blyth$ cat ${OPTICKS_PREFIX}_externals/custom4/0.1.9/lib/Custom4-0.1.9/Custom4Config.cmake | grep WITH
    ./sysrap/SGLFW_tests/CMakeLists.txt:#include_directories($ENV{OPTICKS_PREFIX}/include/SysRap)

    ./CSG_GGeo/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./GeoChain/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./c4/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./examples/UseFindOpticks/CMakeLists.txt:if(DEFINED ENV{OPTICKS_PREFIX})
    ./examples/UseFindOpticks/CMakeLists.txt:    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : NOT Looking for Opticks AS OPTICKS_PREFIX NOT DEFINED")
    ./examples/UseOpticks/CMakeLists.txt:if(DEFINED ENV{OPTICKS_PREFIX})
    ./examples/UsePLogChained/CMakeLists.txt:    $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/plog/include>
    ./npy/CMakeLists.txt:target_include_directories( ${name} PUBLIC ${OPTICKS_PREFIX}/externals/include )

    A[blyth@localhost opticks]$ 



CSG
~~~~

HUH : looks similar, but no hard-coding in generated .cmake, fix anyhow

CSGOptiX
~~~~~~~~~

Similar fix. 


CSG not fixed
~~~~~~~~~~~~~~~~

::

      Imported target "Opticks::G4CX" includes non-existent path

        "/data1/blyth/local/opticks_Debug/include/CSG"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.




Look harder
--------------

::

    A[blyth@localhost cmake]$ pwd
    /data1/blyth/local/opticks_Debug/lib64/cmake
    A[blyth@localhost cmake]$ find . -name '*.cmake' -exec grep -H opticks_Debug {} \;
    ./sysrap/sysrap-config.cmake:INTERFACE_INCLUDE_DIRECTORIES:/data1/blyth/local/opticks_Debug/externals/plog/include
 
    ./csgoptix/csgoptix-targets.cmake:  
        INTERFACE_INCLUDE_DIRECTORIES 
            "${_IMPORT_PREFIX}/externals/glm/glm;
             /usr/local/cuda-12.4/include;
             /usr/local/cuda-12.4/include;
             /cvmfs/opticks.ihep.ac.cn/external/OptiX_800/include;
             /data1/blyth/local/opticks_Debug/include/CSG;
             /data1/blyth/local/opticks_Debug/include/CSG;
             /data1/blyth/local/opticks_Debug/externals/glm/glm;
             ${_IMPORT_PREFIX}/include/CSGOptiX"


    A[blyth@localhost cmake]$ 


::

    A[blyth@localhost opticks]$ find . -name CMakeLists.txt -exec grep -H glm/glm {} \;
    ./CSG/CMakeLists.txt:#    ${OPTICKS_PREFIX}/externals/glm/glm
    ./CSG/CMakeLists.txt:        $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
    ./CSG/CMakeLists.txt:        $<INSTALL_INTERFACE:externals/glm/glm>
    ./CSGOptiX/CMakeLists.txt:      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
    ./CSGOptiX/CMakeLists.txt:      $<INSTALL_INTERFACE:externals/glm/glm>


    ./CSG_GGeo/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./GeoChain/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm
    ./c4/CMakeLists.txt:    ${OPTICKS_PREFIX}/externals/glm/glm

    ./examples/UseOptiX7GeometryInstanced/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiX7GeometryInstancedGAS/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiX7GeometryInstancedGASComp/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiX7GeometryModular/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiX7GeometryStandalone/CMakeLists.txt:    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ./examples/UseOptiXGeometryInstancedStandalone/CMakeLists.txt:    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/externals/glm/glm"
    ./examples/UseOptiXGeometryStandalone/CMakeLists.txt:    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/externals/glm/glm"

    ./sysrap/CMakeLists.txt:   $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
    ./sysrap/CMakeLists.txt:   $<INSTALL_INTERFACE:externals/glm/glm>

    A[blyth@localhost opticks]$ 






Investigation
---------------

Culprit sysrap/CMakeLists.txt::

     55 include_directories(
     56     ${OPTICKS_PREFIX}/externals/glm/glm
     57 )


       
The build time prefix is getting baked in when use include_directories ?:: 

    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ find lib64/cmake -type f -exec grep -H opticks_Debug {} \;
    lib64/cmake/sysrap/sysrap-config.cmake:INTERFACE_INCLUDE_DIRECTORIES:/data1/blyth/local/opticks_Debug/externals/plog/include
    lib64/cmake/sysrap/sysrap-targets.cmake:  INTERFACE_INCLUDE_DIRECTORIES "/usr/local/cuda-12.4/include;/data1/blyth/local/opticks_Debug/externals/glm/glm;/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4;${_IMPORT_PREFIX}/include/SysRap"

    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ pwd
    /data1/blyth/local/opticks_Debug/Opticks-v0.3.1/x86_64--gcc11-geant4_10_04_p02-dbg


First hardcoded prefix is CMake commented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


lib64/cmake/sysrap/sysrap-config.cmake::

     01 
      2 # PROJECT_NAME SysRap
      3 # TOPMATTER
      4 
      5 ## SysRap TOPMATTER
      6 
      7 #[=[ TOPMETA PLog
      8 
      9 [Opticks::PLog]
     10 INTERFACE_INCLUDE_DIRECTORIES:/data1/blyth/local/opticks_Debug/externals/plog/include
     11 
     12 #]=]
     13 
     14 
     15 ## end SysRap TOPMATTER
     16 


Second one is causing the issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

lib64/cmake/sysrap/sysrap-targets.cmake::

     49 # Compute the installation prefix relative to this file.
     50 get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
     51 get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
     52 get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
     53 get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
     54 if(_IMPORT_PREFIX STREQUAL "/")
     55   set(_IMPORT_PREFIX "")
     56 endif()
     57 
     58 # Create imported target Opticks::SysRap
     59 add_library(Opticks::SysRap SHARED IMPORTED)
     60 
     61 set_target_properties(Opticks::SysRap PROPERTIES
     62   INTERFACE_COMPILE_DEFINITIONS 
               "WITH_CUSTOM4;
               \$<\$<CONFIG:Debug>:CONFIG_Debug>;
               \$<\$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>;
               \$<\$<CONFIG:Release>:CONFIG_Release>;
               \$<\$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>;
               OPTICKS_SYSRAP;
               WITH_CHILD;
               PLOG_LOCAL;
               RNG_PHILOX;
               \$<\$<CONFIG:Debug>:DEBUG_TAG>;
               \$<\$<CONFIG:Debug>:DEBUG_PIDX>;
               \$<\$<CONFIG:Debug>:DEBUG_PIDXYZ>;
               \$<\$<CONFIG:Release>:PRODUCTION>;
               WITH_STTF;
               WITH_SLOG"

     63   INTERFACE_INCLUDE_DIRECTORIES 
               "/usr/local/cuda-12.4/include;
               /data1/blyth/local/opticks_Debug/externals/glm/glm;
               /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4;
               ${_IMPORT_PREFIX}/include/SysRap"

     64   INTERFACE_LINK_LIBRARIES 
               "/usr/local/cuda-12.4/lib64/libcudart_static.a;
               Threads::Threads;
               dl;
               /usr/lib64/librt.a;
               Opticks::PLog;
               Opticks::OKConf;
               Opticks::NLJSON;
               ssl;
               crypto"

     65 )


After the below fix that becomes::

     63   INTERFACE_INCLUDE_DIRECTORIES 
             "/usr/local/cuda-12.4/include;
             ${_IMPORT_PREFIX}/externals/glm/glm;
             /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4;
             ${_IMPORT_PREFIX}/include/SysRap"



CMake avoid hardcoded path in INTERFACE_INCLUDE_DIRECTORIES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    697 target_include_directories( ${name} PUBLIC
    698    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    699    ${CUDA_INCLUDE_DIRS}
    700    ${OPTICKS_PREFIX}/externals/glm/glm
    701 )
    702 


* https://cmake.org/cmake/help/latest/command/target_include_directories.html


Include directories usage requirements commonly differ between the build-tree
and the install-tree. The BUILD_INTERFACE and INSTALL_INTERFACE generator
expressions can be used to describe separate usage requirements based on the
usage location. Relative paths are allowed within the INSTALL_INTERFACE
expression and are interpreted as relative to the installation prefix. Relative
paths should not be used in BUILD_INTERFACE expressions because they will not
be converted to absolute. For example::

    target_include_directories(mylib PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>
      $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
    )

* https://cmake.org/cmake/help/latest/command/target_include_directories.html#creating-relocatable-packages




