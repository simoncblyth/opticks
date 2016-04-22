find_library( ThrustRap_LIBRARIES 
              NAMES ThrustRap
              PATHS ${OPTICKS_PREFIX}/lib )

#set(ThrustRap_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/ThrustRap")
set(ThrustRap_INCLUDE_DIRS "${OPTICKS_HOME}/numerics/thrustrap")
set(ThrustRap_DEFINITIONS "")

