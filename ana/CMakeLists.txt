cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Ana)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# order of increasing dependency 
set(SCRIPTS

   __init__.py
   enum.py
   num.py
   log.py
   dat.py
   bpath.py
   key.py
   env.py
   OpticksQuery.py
   base.py
   main.py
   nload.py
   proplib.py
)

install(PROGRAMS ${PARENT_DIR}/__init__.py  DESTINATION py/opticks)
install(PROGRAMS ${SCRIPTS}                 DESTINATION py/opticks/ana)

add_subdirectory(tests)

