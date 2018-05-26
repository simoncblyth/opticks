set(CNPY_PREFIX "$ENV{LOCAL_BASE}/env/cnpy")

find_library( CNPY_LIBRARIES 
              NAMES cnpy
              PATHS ${CNPY_PREFIX}/lib )

set(CNPY_INCLUDE_DIRS "${CNPY_PREFIX}/include")
set(CNPY_DEFINITIONS "")


