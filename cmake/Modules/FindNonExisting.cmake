
set(NonExisting_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( NonExisting_LIBRARIES 
              NAMES NonExisting
              PATHS ${NonExisting_PREFIX}/lib )

if(NOT NonExisting_LIBRARIES)
    set(NonExisting_FOUND FALSE)
    set(NonExisting_INCLUDE_DIRS "")
    set(NonExisting_DEFINITIONS "")
else()
    set(NonExisting_FOUND TRUE)
    set(NonExisting_INCLUDE_DIRS "${NonExisting_PREFIX}/include")
    set(NonExisting_DEFINITIONS "")
endif()


