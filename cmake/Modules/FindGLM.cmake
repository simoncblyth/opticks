
# depending only on LOCAL_BASE allows CMAKE_MODULE_PATH use
# of this FindGLM.cmake without any special environment, other
# than LOCAL_BASE

#set(GLM_PREFIX "${OPTICKS_PREFIX}/externals/glm")
set(GLM_PREFIX "${CMAKE_INSTALL_PREFIX}/externals/glm")

# this is needed by odcs- external so gave to 
# used more general prefixing that works both within
# and without of the Opticks CMake

set(GLM_LIBRARIES "")
set(GLM_INCLUDE_DIRS "${GLM_PREFIX}/glm")
set(GLM_DEFINITIONS "")

