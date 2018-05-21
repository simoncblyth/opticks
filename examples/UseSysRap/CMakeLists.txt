cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(name UseSysRap)

project(${name} VERSION 0.1.0)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
include(EchoTarget)


set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(SysRap REQUIRED CONFIG)   

## CONFIG option loads the targets, preventing  
## use of the old variable based cmake/Modules/FindSysRap.cmake

message(STATUS "SysRap_DIR : ${SysRap_DIR} ")


set(SOURCES
   ${name}.cc
)
set(HEADERS
   ${name}.hh
)

add_library(${name} SHARED ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::SysRap)

#add_executable(Test${name} Test${name}.cc)
#target_link_libraries(Test${name} ${name} )
#install(TARGETS Test${name}  DESTINATION lib)


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

echo_target_std(${name})




