#pragma once

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


inline float gmaxf(const float a, const float b)
{
   return a > b ? a : b ; 
}
inline float gminf(const float a, const float b)
{
   return a > b ? b : a ; 
}

inline float gmaxf( const glm::vec3& v )
{
    return gmaxf( gmaxf(v.x, v.y), v.z );
}


inline void gmaxf(glm::vec4& r, const glm::vec4& a, const glm::vec4& b )
{
    r.x = gmaxf( a.x, b.x );
    r.y = gmaxf( a.y, b.y );
    r.z = gmaxf( a.z, b.z );
    r.w = gmaxf( a.w, b.w );
}
inline void gminf(glm::vec4& r, const glm::vec4& a, const glm::vec4& b )
{
    r.x = gminf( a.x, b.x );
    r.y = gminf( a.y, b.y );
    r.z = gminf( a.z, b.z );
    r.w = gminf( a.w, b.w );
}


inline glm::vec4 gminf(const glm::vec4& a, const glm::vec4& b )
{
    glm::vec4 r ; 
    gminf(r, a, b );
    return r ;
}
inline glm::vec4 gmaxf(const glm::vec4& a, const glm::vec4& b )
{
    glm::vec4 r ; 
    gmaxf(r, a, b );
    return r ;
}





#ifdef __clang__

#pragma clang diagnostic pop

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning(pop)

#endif



