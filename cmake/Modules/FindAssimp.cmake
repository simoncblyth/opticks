
#set(Assimp_PREFIX "${OPTICKS_PREFIX}/externals/assimp/assimp")
set(Assimp_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( Assimp_LIBRARIES 
              NAMES assimp
              PATHS ${Assimp_PREFIX}/lib )

set(Assimp_INCLUDE_DIRS "${Assimp_PREFIX}/include")
set(Assimp_DEFINITIONS "")

