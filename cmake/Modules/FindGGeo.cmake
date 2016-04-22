find_library( GGeo_LIBRARIES 
              NAMES GGeo
              PATHS ${OPTICKS_PREFIX}/lib )

#set(GGeo_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/GGeo")
set(GGeo_INCLUDE_DIRS "${OPTICKS_HOME}/optix/ggeo")
set(GGeo_DEFINITIONS "")

