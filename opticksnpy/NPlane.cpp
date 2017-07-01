#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>


#include "NPlane.hpp"
#include "NGLMExt.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

void nplane::pdump(const char* msg) const 
{
    std::cout 
       << std::setw(10) << msg << " "
       << desc() 
       << std::endl 
       ; 
}

std::string nplane::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "nplane"
       << gpresent( "n,w", get() )
       << gpresent( "pip", point_in_plane() )
       ;
    return ss.str() ;
}


float nplane::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;
    float sd = glm::dot(normal(),glm::vec3(q)) - distance_to_origin() ;   
    return complement ? -sd : sd ; 
}


bool nplane::intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect )
{
    glm::vec3 n = normal();
    float d = distance_to_origin();
    float idn = 1.f/glm::dot(ray_direction, n );  // <-- infinite when ray is perpendicular to normal 
    float on = glm::dot(ray_origin, n);
    float t_cand = (d - on)*idn ; 

    bool valid_intersect = t_cand > tmin ; 

    if( valid_intersect )
    {
         isect.x = n.x ; 
         isect.y = n.y ; 
         isect.z = n.z ; 
         isect.w = t_cand ;    
    }
    return valid_intersect  ; 
}


glm::vec3 nplane::point_in_plane() const 
{
    glm::vec3 pip = normal() * distance_to_origin() ;
    return pip ; 
}


glm::vec3 nplane::gseedcenter()
{
    glm::vec3 center = point_in_plane();
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

glm::vec3 nplane::gseeddir()
{
    glm::vec4 dir(normal(),0); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}

glm::vec4 nplane::make_transformed(const glm::mat4& t) const 
{
    glm::vec4 pip( point_in_plane(), 1.f );
    glm::vec4 tpip = t * pip ;     

    glm::vec4 n(normal(), 0.f) ;
    glm::vec4 tn = t * n  ; 
    glm::vec4 tnn = glm::normalize(tn) ; 

    float t_dist_to_origin = glm::dot( tnn, tpip );
    return glm::vec4( tnn.x, tnn.y, tnn.z, t_dist_to_origin );
}


float ndisk::z() const 
{
   return plane.param.f.w ;  
}

void ndisk::dump(const char* msg)
{
    char dmsg[128];
    snprintf(dmsg, 128, "ndisk radius %10.4f %s \n", radius, msg );
    plane.dump(dmsg);
}






