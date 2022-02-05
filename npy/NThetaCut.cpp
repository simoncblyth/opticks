#include <limits>

#include "nmat4triple.hpp"
#include "NThetaCut.hpp"

/**
TODO: implement this  distance to the surface 
**/

float nthetacut::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 q(x_,y_,z_,1.0); 
    if(gtransform) q = gtransform->v * q ;

    float sd = 0.f ;  
 
    return complement ? -sd : sd ;
} 


// without these placeholders get assert during X4Solid conversion with NCSG::Adopt
int nthetacut::par_euler() const { return 0 ;   }
unsigned nthetacut::par_nsurf() const { return 0 ; }
unsigned nthetacut::par_nvertices(unsigned , unsigned ) const { return 0 ; }

