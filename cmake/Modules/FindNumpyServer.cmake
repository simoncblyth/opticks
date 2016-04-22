find_library( NumpyServer_LIBRARIES 
              NAMES NumpyServer
              PATHS ${OPTICKS_PREFIX}/lib )

#set(NumpyServer_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/NumpyServer")
set(NumpyServer_INCLUDE_DIRS "${OPTICKS_HOME}/boost/basio/numpyserver")
set(NumpyServer_DEFINITIONS "")

