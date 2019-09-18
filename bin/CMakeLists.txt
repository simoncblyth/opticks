cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Bin)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

# increasing dependency order
set(SCRIPTS
   o.sh
   tboolean.sh
)

install(PROGRAMS ${SCRIPTS}   DESTINATION bin)

add_subdirectory(tests)

