
set(GGeo_PREFIX "$ENV{LOCAL_BASE}/env/optix/ggeo")

find_library( GGeo_LIBRARIES 
              NAMES GGeo
              PATHS ${GGeo_PREFIX}/lib )

set(GGeo_INCLUDE_DIRS "${GGeo_PREFIX}/include")
set(GGeo_DEFINITIONS "")

