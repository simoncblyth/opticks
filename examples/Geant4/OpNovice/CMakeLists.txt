#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpNovice)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" OFF)

#if(WITH_GEANT4_UIVIS)
#  find_package(Geant4 REQUIRED ui_all vis_all)
#else()
#  find_package(Geant4 REQUIRED)
#endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
#include(${Geant4_USE_FILE})

find_package( G4OK CONFIG REQUIRED )
#echo_target_std(Opticks::G4OK)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
#include_directories(${PROJECT_SOURCE_DIR}/include 
#                    ${Geant4_INCLUDE_DIR})

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(OpNovice OpNovice.cc ${sources} ${headers})
#target_link_libraries(OpNovice ${Geant4_LIBRARIES} )

target_link_libraries(OpNovice Opticks::G4OK )
target_include_directories(OpNovice PRIVATE ${PROJECT_SOURCE_DIR}/include )
target_compile_definitions( OpNovice PRIVATE WITH_OPTICKS )


#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build OpNovice. This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.
#
set(OpNovice_SCRIPTS
    OpNovice.out
    OpNovice.in
    optPhoton.mac
    gui.mac
    vis.mac
  )

foreach(_script ${OpNovice_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
#install(TARGETS OpNovice DESTINATION bin)
install(TARGETS OpNovice DESTINATION lib)

