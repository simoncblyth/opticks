cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseFindOpticks)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

if(DEFINED ENV{OPTICKS_PREFIX})
    set(Opticks_VERBOSE YES)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{HOME}/opticks/cmake/Modules")
    find_package(Opticks MODULE)
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Opticks_FOUND:${Opticks_FOUND}" )
else()
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : NOT Looking for Opticks AS OPTICKS_PREFIX NOT DEFINED")
endif()

find_package(SysRap REQUIRED)


set(SOURCES DemoLib.cc DEMO_LOG.cc)
set(HEADERS DemoLib.cc DEMO_LOG.hh)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )


target_include_directories(${name}
     PUBLIC
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${name} PUBLIC Opticks::SysRap)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


add_subdirectory(tests)





