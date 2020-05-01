cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksCMakeModules)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

include(GNUInstallDirs)
message(STATUS "OpticksCMakeModules.CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")

set(MODS_SKIPPED
    Modules/EnvBuildOptions.cmake
    Modules/FindNonExisting.cmake
    Modules/OpticksConfigureConfigScript.cmake
)

set(MODS
    Modules/EchoTarget.cmake
    Modules/TopMetaTarget.cmake
    Modules/FindG4.cmake
    Modules/FindGLM.cmake
    Modules/FindImGui.cmake
    Modules/FindOpenMesh.cmake
    Modules/FindOptiX.cmake
    Modules/FindOpticksAssimp.cmake
    Modules/FindOpticksCUDA.cmake
    Modules/FindOpticksGLEW.cmake
    Modules/FindOpticksGLFW.cmake
    Modules/FindOpticksXercesC.cmake
    Modules/FindPLog.cmake
    Modules/FindPPM.cmake
    Modules/OpticksBuildOptions.cmake
    Modules/OpticksCUDAFlags.cmake
    Modules/OpticksCXXFlags.cmake
)

install(FILES ${MODS} DESTINATION cmake/Modules)
add_subdirectory(tests)

