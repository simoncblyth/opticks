
set(OpticksOp_PREFIX "$ENV{LOCAL_BASE}/env/opticksop")

find_library( OpticksOp_LIBRARIES 
              NAMES OpticksOp
              PATHS ${OpticksOp_PREFIX}/lib )

set(OpticksOp_INCLUDE_DIRS "${OpticksOp_PREFIX}/include")
set(OpticksOp_DEFINITIONS "")

