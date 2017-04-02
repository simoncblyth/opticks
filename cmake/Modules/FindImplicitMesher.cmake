
set(ImplicitMesher_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( ImplicitMesher_LIBRARIES 
              NAMES XImplicitMesher
              PATHS ${ImplicitMesher_PREFIX}/lib )

if(NOT ImplicitMesher_LIBRARIES)
    set(ImplicitMesher_FOUND FALSE)
    set(ImplicitMesher_INCLUDE_DIRS "")
    set(ImplicitMesher_DEFINITIONS "")
else()
    set(ImplicitMesher_FOUND TRUE)
    set(ImplicitMesher_INCLUDE_DIRS "${ImplicitMesher_PREFIX}/include")
    set(ImplicitMesher_DEFINITIONS "")
endif()


