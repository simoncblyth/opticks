
set(DualContouringSample_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( DualContouringSample_LIBRARIES 
              NAMES DualContouringSample
              PATHS ${DualContouringSample_PREFIX}/lib )

if(NOT DualContouringSample_LIBRARIES)
    set(DualContouringSample_FOUND FALSE)
    set(DualContouringSample_INCLUDE_DIRS "")
    set(DualContouringSample_DEFINITIONS "")
else()
    set(DualContouringSample_FOUND TRUE)
    set(DualContouringSample_INCLUDE_DIRS "${DualContouringSample_PREFIX}/include")
    set(DualContouringSample_DEFINITIONS "")
endif()


