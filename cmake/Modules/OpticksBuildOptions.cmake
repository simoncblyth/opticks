message(STATUS "Configuring ${CMAKE_PROJECT_NAME}")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(STATUS " CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR} ")
   message(STATUS " CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

include(CTest)
#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${CMAKE_PROJECT_NAME}")

find_package(BCM)
include(BCMDeploy)
include(BCMSetupVersion)  # not yet used in anger, see examples/UseGLM
include(EchoTarget)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

include(OpticksCompilationFlags)   




