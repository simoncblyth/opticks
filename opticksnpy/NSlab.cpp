
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "NSlab.hpp"

/*

                     +1

       +1        -----0----------

       0         --- -1 ---------

       -1        -----0----------

                     +1

*/

float nslab::operator()(float x, float y, float z) 
{
    glm::vec4 p(x,y,z,1.0); 
    if(gtransform) p = gtransform->v * p ; 
    float d = glm::dot( glm::vec3(p), normal) ;
    float doff = abs(d) - abs(offset) ; 
    return doff  ;
} 


glm::vec3 nslab::gcenter()
{
    glm::vec3 center(0,0,0);
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}


void nslab::pdump(const char* msg, int verbosity )
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " normal " << normal 
              << " offset " << offset
              << " gtransform? " << !!gtransform
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}








/*

http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

Box - signed - exact

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

*/

