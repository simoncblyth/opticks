cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCUDA)
project(${name} VERSION 0.1.0)

set(BUILD_SHARED_LIBS ON)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
    
include(EchoTarget)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(OpticksCUDA REQUIRED MODULE) # /opt/local/share/cmake-3.11/Modules/FindCUDA.cmake 

set(SOURCES ${name}.cu)
set(HEADERS ${name}.h)

CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

get_target_property(CALILL ${name} INTERFACE_LINK_LIBRARIES )
message(STATUS "${name} CAL.INTERFACE_LINK_LIBRARIES.bef : ${CALILL} ") 

set_target_properties(${name}  PROPERTIES INTERFACE_LINK_LIBRARIES "Opticks::CUDA" )

get_target_property(CALILL ${name} INTERFACE_LINK_LIBRARIES )
message(STATUS "${name} CAL.INTERFACE_LINK_LIBRARIES.aft : ${CALILL} ") 


# CUDA_ADD_LIBRARY sets raw library path and linker args into target prop INTERFACE_LINK_LIBRARIES
# yielding '/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a;-Wl,-rpath,/usr/local/cuda/lib;Opticks::CUDA'
#
# So kludge tidy that to just "Opticks::CUDA" (which is transitively friendly) 
# before exporting the target.
#

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)





