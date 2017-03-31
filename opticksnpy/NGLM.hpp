
#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#elif defined(_MSC_VER)

#pragma warning(push)
// nonstandard extension used: nameless struct/union  (from glm )
#pragma warning( disable : 4201 )
// members needs to have dll-interface to be used by clients
#pragma warning( disable : 4251 )

#endif


#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>  
#include <glm/gtx/quaternion.hpp>  
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>


#ifdef __clang__

#pragma clang diagnostic pop

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning(pop)

#endif



