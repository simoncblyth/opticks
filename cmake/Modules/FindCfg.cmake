find_library( Cfg_LIBRARIES 
              NAMES Cfg
              PATHS ${OPTICKS_PREFIX}/lib )

#set(Cfg_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/Cfg")
set(Cfg_INCLUDE_DIRS "${OPTICKS_HOME}/boost/bpo/bcfg")

set(Cfg_DEFINITIONS "")

