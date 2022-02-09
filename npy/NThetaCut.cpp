#include <limits>

#include "nmat4triple.hpp"
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

nthetacut* nthetacut::make(OpticksCSG_t type )  // static 
{
    nthetacut* n = new nthetacut ; 
    assert( type == CSG_THETACUT || type == CSG_LTHETACUT ); 
    nnode::Init(n,type) ; 
    return n ; 
}

nthetacut* nthetacut::make(OpticksCSG_t type, const nquad& p0, const nquad& p1 )
{
    nthetacut* n = nthetacut::make(type); 
    n->param = p0 ;    
    n->param1 = p1 ;    

    n->pdump("make_thetacut"); 

    return n ; 
}



/**
nthetacut::make : this needs to match CSG/CSGNode::PrepThetaCutParam
----------------------------------------------------------------------

The input parameters: startTheta, deltaTheta configure two polar angles in range [0., 1.] (units of pi)

* startTheta
* startTheta+deltaTheta

A do nothing thetacut "cutter" would have have startTheta=0. deltaTheta=1. 


           Z

           0.0
       \   :   /
        \  :  /
         \ : /
          \:/ 
           O - - -  0.5    X
          /:\
         / : \
        /  :  \
       /   :   \
           1.0


+---------------------------+----------------------+---------------------------------+-------------------------------+
| (startTheta, deltaTheta)  | (theta0, theta1)     |  Notes                          |  Cutaway                      |
+===========================+======================+=================================+===============================+
|  (0.0, 1.0)               |  (0.0, 1.0)          |  Full range of theta            |                               |
+---------------------------+----------------------+---------------------------------+-------------------------------+
|  (0.1, 0.9)               |  (0.1, 1.0)          |  top hole around zenith         |  0.0->0.1                     |
+---------------------------+----------------------+---------------------------------+-------------------------------+
|  (0.0, 0.9)               |  (0.0, 0.9)          |  bot hole around nadir          |  0.9->1.0                     |
+---------------------------+----------------------+---------------------------------+-------------------------------+
|  (0.25, 0.50)             |  (0.25, 0.75)        |  butterfly or bowtie            |  0.0->0.25, 0.75->1.0         |
+---------------------------+----------------------+---------------------------------+-------------------------------+
|  (0.45, 0.10)             |  (0.45, 0.55)        |  twisted belt                   |  0.0->0.45, 0.55->1.0         |
+---------------------------+----------------------+---------------------------------+-------------------------------+

**/




nthetacut* nthetacut::make(OpticksCSG_t type, double startTheta_pi, double deltaTheta_pi )
{
    double theta0_pi = startTheta_pi  ;
    double theta1_pi = startTheta_pi + deltaTheta_pi ;
    assert( theta0_pi >= 0. && theta0_pi <= 1.) ; 
    assert( theta1_pi >= 0. && theta1_pi <= 1.) ; 
 
    bool has_thetacut = theta0_pi > 0. || theta1_pi < 1. ; 
    assert(has_thetacut);  

    const double pi = glm::pi<double>() ; 

    double theta0 = theta0_pi*pi ; 
    double theta1 = theta1_pi*pi ; 

    double cosTheta0 = std::cos(theta0);
    double sinTheta0 = std::sin(theta0);
    double cosTheta0Sign = cosTheta0/std::abs(cosTheta0) ; 
    double tanTheta0 = std::tan(theta0); 
    double tan2Theta0 = tanTheta0*tanTheta0 ; 

    double cosTheta1 = std::cos(theta1); 
    double sinTheta1 = std::sin(theta1);
    double cosTheta1Sign = cosTheta1/std::abs(cosTheta1) ; 
    double tanTheta1 = std::tan(theta1); 
    double tan2Theta1 = tanTheta1*tanTheta1 ; 

  
    // following Lucas recommendations from CSG/csg_intersect_leaf_thetacut.h 
    nquad p0, p1 ; 
    p0.f.x = float(theta0_pi == 0.5 ? 0. : cosTheta0Sign ); 
    p0.f.y = float(theta0_pi == 0.5 ? 0. : tan2Theta0 ); 
    p0.f.z = float(theta1_pi == 0.5 ? 0. : cosTheta1Sign ); 
    p0.f.w = float(theta1_pi == 0.5 ? 0. : tan2Theta1 ); 

    p1.f.x = float(cosTheta0) ; 
    p1.f.y = float(sinTheta0) ; 
    p1.f.z = float(cosTheta1) ; 
    p1.f.w = float(sinTheta1) ; 

    nthetacut* n = nthetacut::make(type, p0, p1); 
    return n ; 
}



