find_library( ThrustRap_LIBRARIES 
              NAMES ThrustRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT ThrustRap_LIBRARIES)
       set(ThrustRap_LIBRARIES ThrustRap)
    endif()
endif(SUPERBUILD)


set(ThrustRap_INCLUDE_DIRS "${ThrustRap_SOURCE_DIR}")
set(ThrustRap_DEFINITIONS "")

