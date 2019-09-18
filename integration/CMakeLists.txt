cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Integration)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(SCRIPTS
   tests/tboolean.bash
)

install(FILES integration.bash DESTINATION integration)
install(FILES ${SCRIPTS}       DESTINATION integration/tests)

add_subdirectory(tests)


