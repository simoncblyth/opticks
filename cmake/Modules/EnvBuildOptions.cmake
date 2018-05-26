message(STATUS "Configuring ${name}")

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(EnvCompilationFlags)   




