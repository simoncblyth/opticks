cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksGeometry)
project(${name} VERSION 0.1.0)

include(EnvBuildOptions)
include(EnvCompilationFlags)   

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)

#[==[
              SysRap : BCM PLog 
              OKConf : BCM SysRap OptiX G4 
            BoostRap : BCM Boost PLog SysRap 
                 NPY : BCM GLM PLog OpenMesh SysRap BoostRap YoctoGL ImplicitMesher DualContouringSample 
         OpticksCore : BCM OKConf NPY 
                GGeo : BCM OpticksCore 
           AssimpRap : BCM OpticksAssimp GGeo 
         OpenMeshRap : BCM GGeo OpticksCore 
     OpticksGeometry : OpticksCore AssimpRap OpenMeshRap  

#]==]

find_package(OpticksCore REQUIRED CONFIG)
find_package(AssimpRap   REQUIRED CONFIG)
find_package(OpenMeshRap REQUIRED CONFIG)

set(SOURCES

    OKGEO_LOG.cc
    OpticksGeometry.cc
    OpticksAim.cc
    OpticksHub.cc
    OpticksGen.cc
    OpticksGun.cc
    OpticksIdx.cc

)
set(HEADERS

    OKGEO_LOG.hh
    OKGEO_API_EXPORT.hh
    OKGEO_HEAD.hh
    OKGEO_TAIL.hh

    OpticksGeometry.hh
    OpticksHub.hh
    OpticksGen.hh
    OpticksGun.hh
    OpticksIdx.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_link_libraries( ${name} PUBLIC  Opticks::OpticksCore Opticks::AssimpRap Opticks::OpenMeshRap )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


