cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseRelease)
project(${name} VERSION 0.1.0)

# include(OpticksBuildOptions)  # trying without 

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
   set(CMAKE_CXX_STANDARD 14) 
   set(CMAKE_CXX_STANDARD_REQUIRED on) 
else ()
   set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100 forcing c++17 : BUT that restricts to gcc 5+ requiring 
   set(CMAKE_CXX_STANDARD_REQUIRED on) 
endif ()


if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()

#set(PLog_VERBOSE ON)
find_package( G4CX CONFIG REQUIRED )

set(SOURCES UseRelease.cc)

add_executable(${name} ${name}.cc ${SOURCES} ${HEADERS})

target_link_libraries(${name} Opticks::G4CX )
target_compile_definitions( ${name} 
  PRIVATE 
      WITH_G4CXOPTICKS
 )

install(TARGETS ${name} DESTINATION lib)

