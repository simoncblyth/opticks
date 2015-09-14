
# testing library-ization of combined OptiX/Thrust package 

set(OptiXThrust_PREFIX "$ENV{LOCAL_BASE}/env/graphics/optixthrust")

find_library( OptiXThrust_LIBRARIES 
              NAMES OptiXThrustMinimalLib
              PATHS ${OptiXThrust_PREFIX}/lib )

set(OptiXThrust_INCLUDE_DIRS "${OptiXThrust_PREFIX}/include")
set(OptiXThrust_DEFINITIONS "")

