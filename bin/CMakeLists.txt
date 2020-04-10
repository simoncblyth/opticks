cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Bin)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

# increasing dependency order
set(SCRIPTS
   opticks-release.bash
   opticks-site.bash
   opticks-site-local.bash
   o.sh
   tboolean.sh
   scan.bash

   CTestTestfile.py 
   CMakeModules.py 
   okdist.py  
   oktar.py
)

# top level python needs to go in bin, 
# but modules that need to be imported must go to py/opticks/bin


macro(install_symlink filepath sympath)
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
endmacro(install_symlink)


install(PROGRAMS ${SCRIPTS}   DESTINATION bin)
#install_symlink( bin/oc.bash oc )


set(PYSCRIPTS
   __init__.py 
   CMakeLists.py
   dist.py
)

install(PROGRAMS ${PYSCRIPTS}                 DESTINATION py/opticks/bin)


add_subdirectory(tests)

