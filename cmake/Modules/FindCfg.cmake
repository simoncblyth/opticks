
set(Cfg_PREFIX "$ENV{LOCAL_BASE}/env/boost/bpo/bcfg")

find_library( Cfg_LIBRARIES 
              NAMES Cfg
              PATHS ${Cfg_PREFIX}/lib )

set(Cfg_INCLUDE_DIRS "${Cfg_PREFIX}/include")
set(Cfg_DEFINITIONS "")

