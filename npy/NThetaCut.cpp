#include <limits>

#include "scuda.h"
#include "squad.h"

#include "nmat4triple.hpp"
#include "SThetaCut.hh"
#include "NThetaCut.hpp"

/**
TODO: implement this  distance to the surface 
**/


#include "PLOG.hh"

const plog::Severity nthetacut::LEVEL = PLOG::EnvLevel("nthetacut", "DEBUG" ); 


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


void nthetacut::pdump(const char* msg) const 
{
    LOG(LEVEL) << msg ; 
    LOG(LEVEL) << " param  (" << param.f.x  << "," << param.f.y << "," << param.f.z << "," << param.f.w << ") " ; 
    LOG(LEVEL) << " param1 (" << param1.f.x  << "," << param1.f.y << "," << param1.f.z << "," << param1.f.w << ") " ; 

}

/**
nthetacut::make 
------------------

**/

nthetacut* nthetacut::Create(double startTheta_pi, double deltaTheta_pi )
{
    nthetacut* n = new nthetacut ; 
    nnode::Init(n,CSG_THETACUT) ; 

    quad q0, q1 ; 
    SThetaCut::PrepareParam( q0, q1, startTheta_pi, deltaTheta_pi ); 

    n->set_p0(q0); 
    n->set_p1(q1); 
    n->pdump("nthetacut::make"); 

    return n ; 
}

nthetacut* nthetacut::Create(const nquad& p0, const nquad& p1)  // static 
{
    nthetacut* n = new nthetacut ; 
    nnode::Init(n,CSG_THETACUT) ; 
    n->param = p0  ;
    n->param1 = p1  ;
 
    return n ; 
}


