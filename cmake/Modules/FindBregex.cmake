
set(Bregex_PREFIX "$ENV{LOCAL_BASE}/env/boost/bregex")

find_library( Bregex_LIBRARIES 
              NAMES bregex
              PATHS ${Bregex_PREFIX}/lib )

set(Bregex_INCLUDE_DIRS "${Bregex_PREFIX}/include")
set(Bregex_DEFINITIONS "")

