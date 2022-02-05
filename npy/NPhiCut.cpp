#include <limits>

#include "nmat4triple.hpp"
#include "NPhiCut.hpp"


/**
nphicut sdf
------------


       sd < 0 | sd > 0         
              |
              |
             plane
              |
              |      .
              |<---->.
       +- - - | - -  + q
              |      .
              |
       +------+-> normal 
      O    ^  |
           |  |
           distance_to_origin       
              |
              |

Complement-ing flips the normal, and changes the sign of sd 







    
           |
           |    sd1      
           |. . . . . . . . +
           |                : sd0
           |                :
           +------------------------

**/


float nphicut::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 q(x_,y_,z_,1.0); 
    if(gtransform) q = gtransform->v * q ;

    float sd0 = glm::dot(normal(0),glm::vec3(q)) ;   
    float sd1 = glm::dot(normal(1),glm::vec3(q)) ;   

    float sd = std::min( sd0, sd1 ); 
 
    // is the other side handled correctly ?
    // maybe yes : because when one is negative its going to pull 
    // the min -ve too 

    return complement ? -sd : sd ;
} 









int nphicut::par_euler() const 
{
    return 0 ;  
}
unsigned nphicut::par_nsurf() const 
{
    return 0 ;  
}
unsigned nphicut::par_nvertices(unsigned , unsigned ) const 
{
    return 0 ;  
}


