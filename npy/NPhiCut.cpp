#include <limits>

#include "scuda.h"
#include "squad.h"

#include "SPhiCut.hh"
#include "nmat4triple.hpp"
#include "NPhiCut.hpp"
#include "PLOG.hh"


const plog::Severity nphicut::LEVEL = PLOG::EnvLevel("nphicut", "DEBUG"); 



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

glm::vec3 nphicut::normal(int idx) const 
{ 
    const float& cosPhi0 = param.f.x ; 
    const float& sinPhi0 = param.f.y ; 
    const float& cosPhi1 = param.f.z ; 
    const float& sinPhi1 = param.f.w ; 

    return glm::vec3( idx == 0 ? sinPhi0 : -sinPhi1 , idx == 0 ? -cosPhi0 : cosPhi1 , 0.f); 
}


nphicut* nphicut::Create(const nquad& p0)  // static
{
    nphicut* n = new nphicut ; 
    nnode::Init(n,CSG_PHICUT) ; 
    n->param = p0 ; 
    return n ; 
}
nphicut* nphicut::Create(double startPhi_pi, double deltaPhi_pi ) // static
{
    nphicut* n = new nphicut ; 
    nnode::Init(n,CSG_PHICUT) ; 

    quad q0 ; 
    SPhiCut::PrepareParam( q0, startPhi_pi, deltaPhi_pi ); 

    n->set_p0(q0); 
    n->pdump("nphicut::make"); 

    return n ; 
}

void nphicut::pdump(const char* msg) const 
{
    LOG(LEVEL) << msg ; 
    LOG(LEVEL) << " param  (" << param.f.x  << "," << param.f.y << "," << param.f.z << "," << param.f.w << ") " ; 
    LOG(LEVEL) << " param1 (" << param1.f.x  << "," << param1.f.y << "," << param1.f.z << "," << param1.f.w << ") " ; 
}

