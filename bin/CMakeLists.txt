cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Bin)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#[=[
TODO: review the scripts : many are no longer relevant 
#]=]

set(SCRIPTS
   o.sh
   tboolean.sh
   env.sh
   env-l.sh
   snap.sh
   flight.sh
   jflight.sh
   snapscan.sh
   scan.bash
   oc.bash

   find_package.py
   findpkg.py

   pkg_config.py
   pc.py

   ptx.py
   strace.py 
   js.py 
)

set(FORMER_INSTALLED_SCRIPTS
   opticks-release.bash
   opticks-site.bash
   opticks-site-local.bash
)

set(INSTALLED_SCRIPTS

   CTestTestfile.py 
   CTestLog.py  
   CMakeModules.py 
   oktar.py

)

#if(APPLE)
#   list(APPEND SCRIPTS OKTest_macOS_standalone.sh)
#endif()


# top level python needs to go in bin, 
# but modules that need to be imported must go to py/opticks/bin

macro(install_bin_symlink filepath sympath)
    #install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_INSTALL_PREFIX}/${filepath} ${sympath})")
    # create_symlink seems not to be able to create relative symlinks
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_INSTALL_PREFIX}/bin ln -sf ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
endmacro()


include(GNUInstallDirs)
message(STATUS "Bin.CMAKE_INSTALL_BINDIR:${CMAKE_INSTALL_BINDIR}")


install(PROGRAMS ${INSTALLED_SCRIPTS}   DESTINATION bin)

#install_bin_symlink( oc.bash oc )
#install_bin_symlink( oc.bash opticks-config )


set(PYSCRIPTS
   __init__.py 
   CMakeLists.py
)

install(PROGRAMS ${PYSCRIPTS} DESTINATION py/opticks/bin)


add_subdirectory(tests)

