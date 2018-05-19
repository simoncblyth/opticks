
set(YoctoGL_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( YoctoGL_LIBRARIES 
              NAMES YoctoGL
              PATHS ${YoctoGL_PREFIX}/lib )

if(NOT YoctoGL_LIBRARIES)
    set(YoctoGL_FOUND FALSE)
    set(YoctoGL_INCLUDE_DIRS "")
    set(YoctoGL_DEFINITIONS "")
else()
    set(YoctoGL_FOUND TRUE)
    set(YoctoGL_INCLUDE_DIRS "${YoctoGL_PREFIX}/include")
    set(YoctoGL_DEFINITIONS "")
endif()


