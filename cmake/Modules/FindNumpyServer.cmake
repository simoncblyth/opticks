
set(NumpyServer_PREFIX "$ENV{LOCAL_BASE}/env/boost/basio/numpyserver")

find_library( NumpyServer_LIBRARIES 
              NAMES NumpyServer
              PATHS ${NumpyServer_PREFIX}/lib )

set(NumpyServer_INCLUDE_DIRS "${NumpyServer_PREFIX}/include")
set(NumpyServer_DEFINITIONS "")

