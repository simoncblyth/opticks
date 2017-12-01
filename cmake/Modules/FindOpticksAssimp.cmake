
set(OpticksAssimp_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( OpticksAssimp_LIBRARIES 
              NAMES assimp  assimp-vc100-mtd
              PATHS ${OpticksAssimp_PREFIX}/lib )

set(OpticksAssimp_INCLUDE_DIRS "${OpticksAssimp_PREFIX}/include")
set(OpticksAssimp_DEFINITIONS "")

