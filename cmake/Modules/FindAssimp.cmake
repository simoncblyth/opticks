
#set(Assimp_PREFIX "${OPTICKS_PREFIX}/externals/assimp/assimp")
set(Assimp_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( Assimp_LIBRARIES 
              NAMES assimp  assimp-vc100-mtd
              PATHS ${Assimp_PREFIX}/lib )

set(Assimp_INCLUDE_DIRS "${Assimp_PREFIX}/include")
set(Assimp_DEFINITIONS "")

