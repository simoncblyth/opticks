find_library( NumpyServer_LIBRARIES 
              NAMES NumpyServer
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT NumpyServer_LIBRARIES)
       set(NumpyServer_LIBRARIES NumpyServer)
    endif()
endif(SUPERBUILD)



set(NumpyServer_INCLUDE_DIRS "${NumpyServer_SOURCE_DIR}")
set(NumpyServer_DEFINITIONS "")

