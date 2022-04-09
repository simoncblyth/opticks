CMake CUDA
============

Adding sysrap/SU.cu and sysrap/tests/SUTest.cc ran into CUDA CMake
config issues. Note that the standalone approach in sysrap/tests/SUTest.sh 
worked fine.  


Switching from the ordinary "add_library" to CUDA_ADD_LIBRARY::

    307 CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
    308 
    309 
    310 #add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )


Causes a problem for all the "target_link_libraries" in sysrap/CMakeLists.txt
as CUDA_ADD_LIBRARY internally uses a "plain signature".

::

    -- SYSRAP.NLJSON_FOUND
    CMake Error at CMakeLists.txt:327 (target_link_libraries):
      The plain signature for target_link_libraries has already been used with
      the target "SysRap".  All uses of target_link_libraries with a target must
      be either all-keyword or all-plain.

      The uses of the plain signature are here:

       * /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake:1848 (target_link_libraries)
       * CMakeLists.txt:322 (target_link_libraries)


Workaround is to remove the PUBLIC on the all the target_link_libraries in sysrap/CMakeLists.txt::

    -target_link_libraries( ${name} PUBLIC Opticks::PLog Opticks::OKConf )
    +#target_link_libraries( ${name} PUBLIC Opticks::PLog Opticks::OKConf )
    +target_link_libraries( ${name} Opticks::PLog Opticks::OKConf )
     





