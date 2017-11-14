OXRAP_OPTIX_VERSION
======================


Playing around with selecting an OptiX version with *opticks-cmake-modify-ex3*
leads to a confused cmake build, the header parsing yielding 411. But the 
version 380.

This may be due to another level of caching.

::

     12 set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")
     13 set(OptiX_INSTALL_DIR "/tmp"             CACHE PATH   "Path to OptiX installed location.")
     14 set(COMPUTE_CAPABILITY "0"               CACHE STRING "GPU Compute Capability eg one of 0,30,50,52 " )
     15


Conclude that it is better not to try to modify a cmake configuration to change OptiX version, 
instead just wiping the build dir and doing the primary opticks-cmake is the best way to operate.



primary cmake
-------------------

::

     476 opticks-cmake(){
     477    local msg="=== $FUNCNAME : "
     478    local iwd=$PWD
     479    local bdir=$(opticks-bdir)
     480 
     481    echo $msg configuring installation
     482 
     483    mkdir -p $bdir
     484    [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to wipe build dir and re-configure && return
     485 
     486    opticks-bcd
     487 
     488    g4-
     489    xercesc-
     490 
     491    opticks-cmake-info
     492 
     493    cmake \
     494         -G "$(opticks-cmake-generator)" \
     495        -DCMAKE_BUILD_TYPE=Debug \
     496        -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
     497        -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     498        -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
     499        -DGeant4_DIR=$(g4-cmake-dir) \
     500        -DXERCESC_LIBRARY=$(xercesc-library) \
     501        -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
     502        $* \
     503        $(opticks-sdir)
     504 
     505    cd $iwd
     506 }



cmake modification of fundamentals not advisable
---------------------------------------------------

::

     536 opticks-cmake-modify-ex3(){
     537 
     538   local msg="=== $FUNCNAME : "
     539   local bdir=$(opticks-bdir)
     540   local bcache=$bdir/CMakeCache.txt
     541   [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return
     542   opticks-bcd
     543 
     544   echo $msg opticks-cmakecache-vars BEFORE MODIFY 
     545   opticks-cmakecache-vars
     546 
     547   cmake \
     548        -DOptiX_INSTALL_DIR=/Developer/OptiX_380 \
     549        -DCOMPUTE_CAPABILITY=30 \
     550           . 
     551 
     552   echo $msg opticks-cmakecache-vars AFTER MODIFY 
     553   opticks-cmakecache-vars
     554 
     555 }


Use of detected version
-------------------------

::

    simon:opticks blyth$ opticks-find OXRAP_OPTIX_VERSION
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101 
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101
    ./optickscore/tests/OpticksBufferSpecTest.cc:    LOG(info) << "OXRAP_OPTIX_VERSION : " << OXRAP_OPTIX_VERSION ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION >= 3080
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION >= 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION >= 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION == 3080
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 3090
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3090 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 40000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 40000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 400000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION == 3080,3090,4000,400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 40000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 40000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 400000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION == 3080,3090,4000,400000 : " << OXRAP_OPTIX_VERSION  ;
    ./CMakeLists.txt:       message(STATUS "${name}.OXRAP_OPTIX_VERSION : ${OXRAP_OPTIX_VERSION} ")
    ./CMakeLists.txt:# collects version defines, currently only OXRAP_OPTIX_VERSION and CFG4_G4VERSION_NUMBER
    ./CMakeLists.txt:message("top.OXRAP_OPTIX_VERSION ${OXRAP_OPTIX_VERSION} ")
    ./optixrap/CMakeLists.txt:set(OXRAP_OPTIX_VERSION 0 PARENT_SCOPE)
    ./optixrap/CMakeLists.txt:            set(OXRAP_OPTIX_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
    simon:opticks blyth$ 





OptiX Version Detection
------------------------

OptiX Version detection parses ${OptiX_INCLUDE_DIRS}/optix.h


::

     10 find_package(OptiX ${OPTICKS_OPTIX_VERSION} REQUIRED)
     11 
     12 
     13 ###### Find #define OPTIX_VERSION by parsing optix.h set variable at parent scope 
     14 #
     15 # OptiX not playing ball with CMake version conventions, even EXACT has no teeth, 
     16 # so parse the optix.h header to get the #define OPTIX_VERSION into CMake variable
     17 # This means can know the version at configure time
     18 # the value is written to inc/OpticksCMakeConfig.hh by top level CMakeLists.txt
     19 
     20 set(OXRAP_OPTIX_VERSION 0 PARENT_SCOPE)
     21 if(OptiX_FOUND)
     22    message(STATUS "${name}.OPTICKS_OPTIX_VERSION : ${OPTICKS_OPTIX_VERSION}  FOUND ")
     23    message(STATUS "${name}.OptiX_INCLUDE_DIRS    : ${OptiX_INCLUDE_DIRS} ")
     24    message(STATUS "${name}.OptiX_LIBRARIES       : ${OptiX_LIBRARIES} ")
     25    file(READ "${OptiX_INCLUDE_DIRS}/optix.h" _contents)
     26    #message(STATUS "${name}.contents : ${_contents} ")
     27    string(REGEX REPLACE "\n" ";" _contents "${_contents}")
     28    foreach(_line ${_contents})
     29         if (_line MATCHES "#define OPTIX_VERSION ([0-9]+) ")
     30             set(OXRAP_OPTIX_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
     31             message(STATUS "${name}._line ${_line} ===> ${CMAKE_MATCH_1} ")
     32         endif()
     33    endforeach()
     34 else(OptiX_FOUND)
     35    message(STATUS "${name}.OPTICKS_OPTIX_VERSION : ${OPTICKS_OPTIX_VERSION}  NOT-FOUND ")
     36 endif(OptiX_FOUND)


