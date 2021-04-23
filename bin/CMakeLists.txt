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
   env.sh
   env-l.sh
   snap.sh
   flight.sh
   jflight.sh
   snap_emm.sh
   scan.bash
   oc.bash

   CTestTestfile.py 
   CTestLog.py  
   CMakeModules.py 
   okdist.py  
   oktar.py

   find_package.py
   pkg_config.py
   findpkg.py
   pc.py
   ptx.py
   strace.py 
   js.py 
)

if(APPLE)
   list(APPEND SCRIPTS OKTest_macOS_standalone.sh)
endif()


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


install(PROGRAMS ${SCRIPTS}   DESTINATION bin)
install_bin_symlink( oc.bash oc )
install_bin_symlink( oc.bash opticks-config )


set(PYSCRIPTS
   __init__.py 
   CMakeLists.py
   dist.py
)

install(PROGRAMS ${PYSCRIPTS}                 DESTINATION py/opticks/bin)


add_subdirectory(tests)

