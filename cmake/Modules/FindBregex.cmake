find_library( Bregex_LIBRARIES 
              NAMES Bregex
              PATHS ${OPTICKS_PREFIX}/lib )

#set(Bregex_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/Bregex")
set(Bregex_INCLUDE_DIRS "${OPTICKS_HOME}/boost/bregex")

set(Bregex_DEFINITIONS "")

