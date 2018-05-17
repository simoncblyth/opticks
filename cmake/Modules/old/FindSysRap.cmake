find_library( SysRap_LIBRARIES 
              NAMES SysRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT SysRap_LIBRARIES)
       set(SysRap_LIBRARIES SysRap)
    endif()
endif(SUPERBUILD)

# find_package normally yields NOTFOUND
# when no lib is found at configure time : ie when cmake is run
# but here if SUPERBUILD is defined BRegex_LIBRARIES
# is set to BRegex the target name, 
# which will allow the build to succeed if the target
# is included amongst the add_subdirectory of the super build

set(SysRap_INCLUDE_DIRS "${SysRap_SOURCE_DIR}")
set(SysRap_DEFINITIONS "")




