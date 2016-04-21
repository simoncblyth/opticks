set(ThrustRap_PREFIX "${OPTICKS_PREFIX}/numerics/ThrustRap")

find_library( ThrustRap_LIBRARIES 
              NAMES ThrustRap
              PATHS ${ThrustRap_PREFIX}/lib )

set(ThrustRap_INCLUDE_DIRS "${ThrustRap_PREFIX}/include")
set(ThrustRap_DEFINITIONS "")

