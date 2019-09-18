cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Analytic)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

set(SCRIPTS
   __init__.py
   glm.py 
   prism.py
   textgrid.py
   tbool.py
   nnode_test_cpp.py
   csg.py
)

install(PROGRAMS ${SCRIPTS}  DESTINATION py/opticks/analytic )

add_subdirectory(tests)

