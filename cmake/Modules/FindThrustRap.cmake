
set(ThrustRap_PREFIX "$ENV{LOCAL_BASE}/env/numerics/ThrustRap")

find_library( ThrustRap_LIBRARIES 
              NAMES ThrustRap
              PATHS ${ThrustRap_PREFIX}/lib )

set(ThrustRap_INCLUDE_DIRS "${ThrustRap_PREFIX}/include")
set(ThrustRap_DEFINITIONS "")

