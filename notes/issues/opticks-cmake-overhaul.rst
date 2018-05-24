opticks-cmake-overhaul
=========================

motivation
-----------

Ease of use : make applying Opticks to a Geant4 example (eg LXe) straightforward to 

* configure
* code 

  * TODO : general purpose high level steering 


stages
--------

* DONE : bcm wave thru the projects
* move almost everything from top level CMakeLists.txt into sub-projects : aiming 
  for top level to just be a list of sub-projects 

  * OptiX, G4 detection/parsing moved into OKConf
  * TODO: move opticks-config generation into OKConf too 



ctesting 
-----------

Issues 

* are down from 300 tests to 192
* 20/192 fails, including all thrap-


::

    90% tests passed, 20 tests failed out of 192

    Total Test time (real) = 147.19 sec

    The following tests FAILED:
        120 - GGeoTest.GSceneTest (Child aborted)
        129 - ThrustRapTest.CBufSpecTest (Child aborted)
        130 - ThrustRapTest.TBufTest (Child aborted)
        131 - ThrustRapTest.TRngBufTest (Child aborted)
        132 - ThrustRapTest.expandTest (Child aborted)
        133 - ThrustRapTest.iexpandTest (Child aborted)
        134 - ThrustRapTest.issue628Test (Child aborted)
        135 - ThrustRapTest.printfTest (Child aborted)
        136 - ThrustRapTest.repeated_rangeTest (Child aborted)
        137 - ThrustRapTest.strided_rangeTest (Child aborted)
        138 - ThrustRapTest.strided_repeated_rangeTest (Child aborted)
        139 - ThrustRapTest.float2intTest (Child aborted)
        140 - ThrustRapTest.thrust_curand_estimate_pi (Child aborted)
        141 - ThrustRapTest.thrust_curand_printf (Child aborted)
        142 - ThrustRapTest.thrust_curand_printf_redirect (Child aborted)
        143 - ThrustRapTest.thrust_curand_printf_redirect2 (Child aborted)
        176 - CFG4Test.CTestDetectorTest (Child aborted)
        179 - CFG4Test.CG4Test (Child aborted)
        187 - CFG4Test.CInterpolationTest (Child aborted)
        192 - CFG4Test.CRandomEngineTest (Child aborted)
    Errors while running CTest
    epsilon:build blyth$ 




oxrap
-------

::

    /Users/blyth/opticks-cmake-overhaul
     10               OKConf : BCM OptiX G4  
     20               SysRap : BCM PLog  
     30             BoostRap : BCM Boost PLog SysRap  
     40                  NPY : BCM GLM PLog OpenMesh SysRap BoostRap YoctoGL ImplicitMesher DualContouringSample  
     50          OpticksCore : BCM OKConf NPY  
     60                 GGeo : BCM OpticksCore  
     70            AssimpRap : BCM OpticksAssimp GGeo  
     80          OpenMeshRap : BCM GGeo OpticksCore  

     90      OpticksGeometry : BCM OpticksCore AssimpRap OpenMeshRap  
    100              CUDARap : BCM SysRap OpticksCUDA  
    110            ThrustRap : BCM OpticksCore CUDARap  
    120             OptiXRap : 
    
     OptiX OpticksGeometry ThrustRap 


::

    099 function(optixthrust_add_library target_name)
    100 
    101     # split arguments into four lists 
    102     #  hmm have two different flavors of .cu
    103     #  optix programs to be made into .ptx  
    104     #  and thrust or CUDA non optix sources need to be compiled into .o for linkage
    105 
    106     OPTIXTHRUST_GET_SOURCES_AND_OPTIONS(optix_source_files non_optix_source_files cmake_options options ${ARGN})
    107 
    108     #message( "OPTIXTHRUST:optix_source_files= " "${optix_source_files}" )  
    109     #message( "OPTIXTHRUST:non_optix_source_files= "  "${non_optix_source_files}" )  
    110 
    111     # Create the rules to build the OBJ from the CUDA files.
    112     #message( "OPTIXTHRUST:OBJ options = " "${options}" )  
    113     CUDA_WRAP_SRCS( ${target_name} OBJ non_optix_generated_files ${non_optix_source_files} ${cmake_options} OPTIONS ${options} )
    114 
    115     # Create the rules to build the PTX from the CUDA files.
    116     #message( "OPTIXTHRUST:PTX options = " "${options}" )  
    117     CUDA_WRAP_SRCS( ${target_name} PTX optix_generated_files ${optix_source_files} ${cmake_options} OPTIONS ${options} )
    118 
    119     add_library(${target_name}
    120         ${optix_source_files}
    121         ${non_optix_source_files}
    122         ${optix_generated_files}
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ surely these are just ignored ?

    123         ${non_optix_generated_files}
    124         ${cmake_options}
    125     )
    126 
    127     target_link_libraries( ${target_name}
    128         ${LIBRARIES}
    129       )
    130 
    131 endfunction()
    132 




setup the fork : move dev to local clone
-----------------------------------------

Overhauling CMake infrastructure is bound to cause 
build breakage potentially for an extended period, 
so are unwilling to commit/push the CMake related changes.

Instead: 

1. commit/push all unrelated non-breaking changes are willing to, leaving 
   just the CMake related ones::

::

    epsilon:opticks blyth$ hg st .
    M CMakeLists.txt
    M cmake/Templates/opticks-config.in
    M okop/okop.bash
    M opticks.bash
    M opticksnpy/CMakeLists.txt
    M sysrap/CMakeLists.txt
    ? cmake/Modules/FindOpticks.cmake
    ? cmake/Modules/OpticksConfigureCMakeHelpers.cmake
    ? cmake/Templates/OpticksConfig.cmake.in
    ? examples/FindOpticks/CMakeLists.txt
    ? examples/FindOpticks/FindOpticks.cc
    ? examples/FindOpticks/README.rst
    ? examples/FindOpticks/go.sh
    ? examples/UseNPY/CMakeLists.txt
    ? examples/UseNPY/UseNPY.cc
    ? examples/UseNPY/go.sh
    ? examples/UseSysRap/CMakeLists.txt
    ? examples/UseSysRap/UseSysRap.cc
    ? examples/UseSysRap/go.sh
    epsilon:opticks blyth$ 

2. make a local clone::

    cd ; hg clone opticks opticks-cmake-overhaul    ## apparently this uses hardlinks

    epsilon:opticks-cmake-overhaul blyth$ hg paths -v    ## can pull/update from "mainline" into the overhaul clone 
    default = /Users/blyth/opticks

    epsilon:opticks blyth$ mv examples ../opticks-cmake-overhaul/


building the local clone
---------------------------

::

    sudo mkdir /usr/local/opticks-cmake-overhaul
    sudo chown blyth:staff /usr/local/opticks-cmake-overhaul

Share externals from the standard opticks::

    epsilon:opticks-cmake-overhaul blyth$ cd /usr/local/opticks-cmake-overhaul
    epsilon:opticks-cmake-overhaul blyth$ ln -s ../opticks/externals externals



examples/UseUseBoost failing 
-----------------------------------------

::

    ====== tgt:Opticks::UseBoost tgt_DIR: ================
    tgt='Opticks::UseBoost' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='/usr/local/opticks-cmake-overhaul/include/UseBoost' 

    tgt='Opticks::UseBoost' prop='INTERFACE_LINK_LIBRARIES' defined='0' set='1' value='Boost::filesystem' 

    tgt='Opticks::UseBoost' prop='IMPORTED_CONFIGURATIONS' defined='0' set='1' value='DEBUG' 

    tgt='Opticks::UseBoost' prop='IMPORTED_LOCATION_DEBUG' defined='0' set='1' value='/usr/local/opticks-cmake-overhaul/lib/libUseBoost.dylib' 

    tgt='Opticks::UseBoost' prop='IMPORTED_SONAME_DEBUG' defined='0' set='1' value='@rpath/libUseBoost.dylib' 


    -- Configuring done
    CMake Error at CMakeLists.txt:14 (add_executable):
      Target "UseUseBoost" links to target "Boost::filesystem" but the target was
      not found.  Perhaps a find_package() call is missing for an IMPORTED
      target, or an ALIAS target is missing?




/usr/local/opticks-cmake-overhaul/lib/cmake/useboost/useboost-config.cmake::

  1 
  2 include(CMakeFindDependencyMacro)
  3 # Library: Boost::filesystem
  4 find_dependency(Boost)
  5 
  6 include("${CMAKE_CURRENT_LIST_DIR}/useboost-targets.cmake")
  7 include("${CMAKE_CURRENT_LIST_DIR}/properties-useboost-targets.cmake")



The above looks lacking need to pass component to find_dependency ?

Suspect cause of issue is integrating with targets that are not compliant to the BCM way ?

::

    epsilon:UseBoost blyth$ port contents cmake | grep Boost
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
      /opt/local/share/cmake-3.11/Help/module/FindBoost.rst
      /opt/local/share/cmake-3.11/Modules/FindBoost.cmake
      /opt/local/share/doc/cmake/html/_sources/module/FindBoost.rst.txt
      /opt/local/share/doc/cmake/html/module/FindBoost.html
    epsilon:UseBoost blyth$ 




::

    epsilon:usesysrap blyth$ cat usesysrap-config.cmake

    include(CMakeFindDependencyMacro)
    # Library: Opticks::SysRap
    find_dependency(SysRap)

    include("${CMAKE_CURRENT_LIST_DIR}/usesysrap-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/properties-usesysrap-targets.cmake")



Filesystem gets lost?::

    epsilon:useboost blyth$ cat useboost-config.cmake 

    include(CMakeFindDependencyMacro)
    # Library: Boost::filesystem
    find_dependency(Boost)

    include("${CMAKE_CURRENT_LIST_DIR}/useboost-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/properties-useboost-targets.cmake")
    epsilon:useboost blyth$ 


The generator of that BCMExport.cmake::

     76     if(PARSE_TARGETS)
     77         # Add dependencies
     78         foreach(TARGET ${PARSE_TARGETS})
     79             get_property(TARGET_LIBS TARGET ${TARGET} PROPERTY INTERFACE_LINK_LIBRARIES)
     80             foreach(LIB ${TARGET_LIBS})
     81                 bcm_get_target_package_source(PKG_SRC ${LIB})
     82                 set(HAS_PKG_SRC "$<BOOL:${PKG_SRC}>")
     83                 string(APPEND CONFIG_FILE_CONTENT "# $<$<NOT:${HAS_PKG_SRC}>:Skip >Library: ${LIB}\n")
     84                 string(APPEND CONFIG_FILE_CONTENT "$<${HAS_PKG_SRC}:find_dependency(${PKG_SRC})>\n")
     85             endforeach()
     86         endforeach()


::

     04 function(bcm_get_target_package_source OUT_VAR TARGET)
      5     set(RESULT)
      6     if(TARGET ${TARGET})
      7         get_property(TARGET_ALIAS TARGET ${TARGET} PROPERTY ALIASED_TARGET)
      8         if(TARGET_ALIAS)
      9             set(TARGET ${TARGET_ALIAS})
     10         endif()
     11         get_property(TARGET_IMPORTED TARGET ${TARGET} PROPERTY IMPORTED)
     12         if(TARGET_IMPORTED OR TARGET_ALIAS)
     13             get_property(TARGET_FIND_PACKAGE_NAME TARGET ${TARGET} PROPERTY INTERFACE_FIND_PACKAGE_NAME)
     14             if(NOT TARGET_FIND_PACKAGE_NAME)
     15                 message(SEND_ERROR "The target ${TARGET_FIND_PACKAGE_NAME} does not have information about find_package() call.")
     16             endif()
     17             set(PKG_NAME ${TARGET_FIND_PACKAGE_NAME})
     18             get_property(TARGET_FIND_PACKAGE_VERSION TARGET ${TARGET} PROPERTY INTERFACE_FIND_PACKAGE_VERSION)
     19             if(TARGET_FIND_PACKAGE_VERSION)
     20                 set(PKG_NAME "${PKG_NAME} ${TARGET_FIND_PACKAGE_VERSION}")
     21             endif()
     22             get_property(TARGET_FIND_PACKAGE_EXACT TARGET ${TARGET} PROPERTY INTERFACE_FIND_PACKAGE_EXACT)
     23             if(TARGET_FIND_PACKAGE_EXACT)
     24                 set(PKG_NAME "${PKG_NAME} ${TARGET_FIND_PACKAGE_EXACT}")
     25             endif()
     26             set(RESULT "${PKG_NAME}")
     27             # get_property(TARGET_FIND_PACKAGE_REQUIRED TARGET ${TARGET} PROPERTY INTERFACE_FIND_PACKAGE_REQUIRED)
     28             # get_property(TARGET_FIND_PACKAGE_QUIETLY TARGET ${TARGET} PROPERTY INTERFACE_FIND_PACKAGE_QUIETLY)
     29         endif()
     30     else()
     31         if("${TARGET}" MATCHES "::")
     32             set(TARGET_NAME "$<TARGET_PROPERTY:${TARGET},ALIASED_TARGET>")
     33         else()
     34             set(TARGET_NAME "${TARGET}")
     35         endif()
     36         bcm_shadow_exists(HAS_TARGET ${TARGET})
     37         set(RESULT "$<${HAS_TARGET}:$<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_FIND_PACKAGE_NAME>>")
     38     endif()
     39     set(${OUT_VAR} "${RESULT}" PARENT_SCOPE)
     40 endfunction()



Ahha, some of those properties are not standard CMake, they are defined by BCM::

    epsilon:cmake blyth$ grep define_property *.*
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_FIND_PACKAGE_NAME"
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_FIND_PACKAGE_REQUIRED"
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_FIND_PACKAGE_QUIETLY"
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_FIND_PACKAGE_EXACT"
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_FIND_PACKAGE_VERSION"
    BCMFuture.cmake:define_property(TARGET PROPERTY "INTERFACE_TARGET_EXISTS"
    BCMPkgConfig.cmake:define_property(TARGET PROPERTY "INTERFACE_DESCRIPTION"
    BCMPkgConfig.cmake:define_property(TARGET PROPERTY "INTERFACE_URL"
    BCMPkgConfig.cmake:define_property(TARGET PROPERTY "INTERFACE_PKG_CONFIG_REQUIRES"
    BCMProperties.cmake:    define_property(${scope} PROPERTY "CXX_EXCEPTIONS" INHERITED
    BCMProperties.cmake:    define_property(${scope} PROPERTY "CXX_RTTI" INHERITED
    BCMProperties.cmake:    define_property(${scope} PROPERTY "CXX_STATIC_RUNTIME" INHERITED
    BCMProperties.cmake:    define_property(${scope} PROPERTY "CXX_WARNINGS" INHERITED
    BCMProperties.cmake:    define_property(${scope} PROPERTY "CXX_WARNINGS_AS_ERRORS" INHERITED
    BCMTest.cmake:  define_property(${scope} PROPERTY "ENABLE_TESTS" INHERITED
    BCMTest.cmake:    define_property(${scope} PROPERTY "BCM_TEST_DEPENDENCIES" INHERITED
    epsilon:cmake blyth$ 

