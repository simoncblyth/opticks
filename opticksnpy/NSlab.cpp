
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "NSlab.hpp"

/*

Consider CSG difference of near and far halfspaces

        difference(near-half-space,far-half-space) = max( dnear, -dfar )



                         ^      
                         |
                  -------+---------   
                                  -ve: inside far halfspace
                                      
                         ^       
                         |         
                  -------+---------   
                                  -ve: inside near halfspace
                        
 
                         .  origin


  CSG difference "max(l,-r)" near and far:

           far-halfspace - near-halfspace  = slab 

  CSG intersect "max(l,r)" near and -far (complement of far)


  Not so easy to imagine CSG difference that leaves you with 
  negative space, so instead just flip the order and negate

     A - B =>  -(B - A)

   So you get everything not in the slab.

  View the planes as a half-spaces want to form CSG intersection


*/

float nslab::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;
 
    float d = glm::dot(n, glm::vec3(q)) ; // distance from q to the nearest point on plane thru origin
    float da =  d - a ;             
    float db =  d - b  ;   // b > a

    //  CSG 
    //     union(l,r)     ->  min(l,r)
    //     intersect(l,r) ->  max(l,r)
    //     difference(l,r) -> max(l,-r)
    //     
    float sd = fmaxf(db, -da)   ;

    return complement ? -sd : sd ; 
} 


/*

Ray
     q = ray_origin + t * ray_direction 

Planes
     q.n = far  
     q.n = near     "-near" for outward slab normals? 
 
Intersects
      q.n = ray_origin.n + t * ray_direction.n 
      t = (q.n - ray_origin.n) / ray_direction.n

Just because the ray intersects the slab doesnt mean its valid, 
there are 3 possibilities depending in where t_min lies wrt t_near and t_far.

                         t_near       t_far   
                           |           |
                 -----1----|----2------|------3---------->
                           |           |

*/


bool nslab::intersect( const float t_min, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect )
{
    float idn = 1.f/glm::dot(ray_direction, n );  // <-- infinite when ray is perpendicular to normal 
    float on = glm::dot(ray_origin, n );

    float ta = (a - on)*idn ; 
    float tb = (b - on)*idn ; 

    float t_near = fminf(ta,tb);  // order the intersects 
    float t_far  = fmaxf(ta,tb);

    float t_cand = t_near > t_min ?  t_near : ( t_far > t_min ? t_far : t_min ) ;

    bool valid_intersect = t_cand > t_min ; 
    bool b_hit = t_cand == tb ;    
     
     std::cout 
         << "nslab::intersect" 
         << " a " << a 
         << " b " << b 
         << " ta " << ta
         << " tb " << tb
         << " idn " << idn
         << " on " << on
         << " t_near " << t_near
         << " t_far " << t_far
         << " t_cand " << t_cand
         << " b_hit " << b_hit 
         << std::endl ; 

    if( valid_intersect )
    {
         isect.x = b_hit ? n.x : -n.x ; 
         isect.y = b_hit ? n.y : -n.y ; 
         isect.z = b_hit ? n.z : -n.z ; 
         isect.w = t_cand ;    
    }
    return valid_intersect  ; 
}




glm::vec3 nslab::gseedcenter()
{
    glm::vec3 center = n * (a+b)/2.f ;
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

glm::vec3 nslab::gseeddir()
{
    glm::vec4 dir(n,0); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}



void nslab::pdump(const char* msg, int verbosity )
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " n " << n
              << " a  " << a
              << " b (b>a) " << b
              << " gtransform? " << !!gtransform
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}




