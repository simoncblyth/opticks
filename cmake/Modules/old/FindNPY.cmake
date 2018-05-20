find_library( NPY_LIBRARIES 
              NAMES NPY
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT NPY_LIBRARIES)
       set(NPY_LIBRARIES NPY)
    endif()
endif(SUPERBUILD)


set(NPY_INCLUDE_DIRS "${NPY_SOURCE_DIR}")

set(NPY_DEFINITIONS "")

