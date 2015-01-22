
set(Assimp_PREFIX "$ENV{LOCAL_BASE}/env/graphics")

find_library( Assimp_LIBRARIES 
              NAMES assimp
              PATHS ${Assimp_PREFIX}/lib )

set(Assimp_INCLUDE_DIRS "${Assimp_PREFIX}/include")
set(Assimp_DEFINITIONS "")

