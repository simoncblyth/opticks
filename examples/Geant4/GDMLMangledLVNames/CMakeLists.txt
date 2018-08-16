cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GDMLMangledLVNames)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OKConf_VERBOSE ON)
find_package( G4OK CONFIG REQUIRED )

set(SOURCES
    DetectorConstruction.cc
)

set(HEADERS
    DetectorConstruction.hh
)

add_executable(${name} ${name}.cc ${SOURCES} ${HEADERS})

target_link_libraries(${name} Opticks::G4OK )
target_compile_definitions( ${name} PRIVATE WITH_OPTICKS )

install(TARGETS ${name} DESTINATION lib)



