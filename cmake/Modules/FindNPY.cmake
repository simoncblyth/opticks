
set(NPY_PREFIX "$ENV{LOCAL_BASE}/env/numerics/npy")

find_library( NPY_LIBRARIES 
              NAMES NPY
              PATHS ${NPY_PREFIX}/lib )

set(NPY_INCLUDE_DIRS "${NPY_PREFIX}/include")
set(NPY_DEFINITIONS "")

