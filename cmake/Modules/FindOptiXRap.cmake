
set(OptiXRap_PREFIX "$ENV{LOCAL_BASE}/env/graphics/OptiXRap")

find_library( OptiXRap_LIBRARIES 
              NAMES OptiXRap
              PATHS ${OptiXRap_PREFIX}/lib )

set(OptiXRap_INCLUDE_DIRS "${OptiXRap_PREFIX}/include")
set(OptiXRap_DEFINITIONS "")

