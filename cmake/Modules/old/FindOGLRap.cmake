find_library( OGLRap_LIBRARIES 
              NAMES OGLRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OGLRap_LIBRARIES)
       set(OGLRap_LIBRARIES OGLRap)
    endif()
endif(SUPERBUILD)


set(OGLRap_INCLUDE_DIRS "${OGLRap_SOURCE_DIR}")
set(OGLRap_DEFINITIONS "")

