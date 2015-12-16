
set(Opticks_PREFIX "$ENV{LOCAL_BASE}/env/opticks")

find_library( Opticks_LIBRARIES 
              NAMES Opticks
              PATHS ${Opticks_PREFIX}/lib )

set(Opticks_INCLUDE_DIRS "${Opticks_PREFIX}/include")
set(Opticks_DEFINITIONS "")

