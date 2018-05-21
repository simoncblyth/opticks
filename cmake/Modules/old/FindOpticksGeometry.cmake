find_library( OpticksGeometry_LIBRARIES 
              NAMES OpticksGeometry
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OpticksGeometry_LIBRARIES)
       set(OpticksGeometry_LIBRARIES OpticksGeometry)
    endif()
endif(SUPERBUILD)

set(OpticksGeometry_INCLUDE_DIRS "${OpticksGeometry_SOURCE_DIR}")

set(OpticksGeometry_DEFINITIONS "")

