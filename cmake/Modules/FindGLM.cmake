
# depending only on LOCAL_BASE allows CMAKE_MODULE_PATH use
# of this FindGLM.cmake without any special environment, other
# than LOCAL_BASE

set(GLM_PREFIX "$ENV{LOCAL_BASE}/opticks/externals/glm")

set(GLM_LIBRARIES "")
set(GLM_INCLUDE_DIRS "${GLM_PREFIX}/glm")
set(GLM_DEFINITIONS "")

