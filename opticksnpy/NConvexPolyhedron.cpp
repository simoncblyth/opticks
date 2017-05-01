
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "NConvexPolyhedron.hpp"


float nconvexpolyhedron::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;

    unsigned num_planes = planes.size(); 
    float dmax = 0.f ; 


   // need to distinguish cases of 
   // being inside and outside the planes

    for(unsigned i=0 ; i < num_planes ; i++)
    {
        const nvec4& plane = planes[i]; 
        glm::vec3 pnorm(plane.x, plane.y, plane.z );
        float pdist = fabsf(plane.w) ;   // <-- assert on this in ctor, directionality from the normal, not sign of dist 

        float d0 = glm::dot(pnorm, glm::vec3(q)) ;   // distance from q to the normal plane thru origin
        if(d0 == 0.f) continue ; 

        // d0 < 0.f  q inside plane halfspace
         
        float d = d0 - pdist ; 

        //float d = d0 < 0.f ? d0 - pdist : -(d0 - pdist) ; 
        //std::cout << " " << std::setw(10) << d ; 

        if(d > dmax) dmax = d ;      
        // assuming origin is inside the convexpolyhedron ? 
    }
    //std::cout << std::endl ; 

    return dmax ; 
} 


bool nconvexpolyhedron::intersect( const float t_min, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect )
{

/*
    float idn = 1.f/glm::dot(ray_direction, n );  // <-- infinite when ray is perpendicular to normal 
    float on = glm::dot(ray_origin, n );

    float ta = (a - on)*idn ; 
    float tb = (b - on)*idn ; 

    float t_near = fminf(ta,tb);  // order the intersects 
    float t_far  = fmaxf(ta,tb);

    float t_cand = t_near > t_min ?  t_near : ( t_far > t_min ? t_far : t_min ) ;

    bool valid_intersect = t_cand > t_min ; 
    bool b_hit = t_cand == tb ;    
     
    if( valid_intersect )
    {
         isect.x = b_hit ? n.x : -n.x ; 
         isect.y = b_hit ? n.y : -n.y ; 
         isect.z = b_hit ? n.z : -n.z ; 
         isect.w = t_cand ;    
    }
    return valid_intersect  ; 

*/
    return false ; 
}




glm::vec3 nconvexpolyhedron::gseedcenter()
{
    glm::vec3 center(0.f,0.f,0.f) ;
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

glm::vec3 nconvexpolyhedron::gseeddir()
{
    glm::vec4 dir(1.,0.,0.,0.); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


void nconvexpolyhedron::pdump(const char* msg, int verbosity )
{
    unsigned num_planes = planes.size();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " gtransform? " << !!gtransform
              << " num_planes " << num_planes
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;



}




