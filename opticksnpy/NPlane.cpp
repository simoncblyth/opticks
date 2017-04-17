#include "NPlane.hpp"
#include "NGLMExt.hpp"
#include "GLMPrint.hpp"
#include <cstring>

void nplane::pdump(const char* msg, int /*verbosity*/)
{
    param.dump(msg);
}

float nplane::operator()(float x, float y, float z) 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;
    return glm::dot(n,glm::vec3(q)) - d ;   
}


bool nplane::intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect )
{
    float idn = 1.f/glm::dot(ray_direction, n );  // <-- infinite when ray is perpendicular to normal 
    float on = glm::dot(ray_origin, n );
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


glm::vec3 nplane::gcenter()
{
    glm::vec3 center = n * d ;
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}





float ndisc::z() const 
{
   return plane.param.f.w ;  
}

void ndisc::dump(const char* msg)
{
    char dmsg[128];
    snprintf(dmsg, 128, "ndisc radius %10.4f %s \n", radius, msg );
    plane.dump(dmsg);
}






