set(CNPY_PREFIX "$ENV{LOCAL_BASE}/env/cnpy")

#set(CNPY_LIBRARIES "${CNPY_PREFIX}/lib/libcnpy.dylib")
#set(CNPY_LIBRARIES "${CNPY_PREFIX}/lib/libcnpy.so")

find_library( CNPY_LIBRARIES 
              NAMES cnpy
              PATHS ${CNPY_PREFIX}/lib )

set(CNPY_INCLUDE_DIRS "${CNPY_PREFIX}/include")
set(CNPY_DEFINITIONS "")


