#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "SThetaCut.hh"

const plog::Severity SThetaCut::LEVEL = PLOG::EnvLevel("SThetaCut", "DEBUG" ); 

/**
SThetaCut::PrepareParam 
------------------------

This is used by

* npy/NThetaCut.cpp
* CSG/CSGNode.cc


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


void SThetaCut::PrepareParam( quad& q0, quad& q1, double startTheta_pi, double deltaTheta_pi )
{
    double theta0_pi = startTheta_pi  ;
    double theta1_pi = startTheta_pi + deltaTheta_pi ;

    bool plane_theta0 = theta0_pi == 0.5 ;   // cone degenerates to a plane for angle 0.5 
    bool plane_theta1 = theta1_pi == 0.5 ; 

    assert( theta0_pi >= 0. && theta0_pi <= 1.) ; 
    assert( theta1_pi >= 0. && theta1_pi <= 1.) ; 
 
    bool has_thetacut = theta0_pi > 0. || theta1_pi < 1. ; 
    assert(has_thetacut);  

    const double pi = M_PIf ; 
    double theta0 = theta0_pi*pi ; 
    double theta1 = theta1_pi*pi ; 

    double cosTheta0 = std::cos(theta0);
    double sinTheta0 = std::sin(theta0);
    double tanTheta0 = std::tan(theta0); 

    double cosTheta1 = std::cos(theta1); 
    double sinTheta1 = std::sin(theta1);
    double tanTheta1 = std::tan(theta1); 

    double cosTheta0Sign = cosTheta0/std::abs(cosTheta0) ; 
    double cosTheta1Sign = cosTheta1/std::abs(cosTheta1) ; 

    double tan2Theta0 = tanTheta0*tanTheta0 ; 
    double tan2Theta1 = tanTheta1*tanTheta1 ; 


    q0.f.x = float(plane_theta0 ? 0. : cosTheta0Sign );  
    q0.f.y = float(plane_theta0 ? 0. : tan2Theta0 );  
    q0.f.z = float(plane_theta1 ? 0. : cosTheta1Sign );  
    q0.f.w = float(plane_theta1 ? 0. : tan2Theta1 );  

    q1.f.x = float(cosTheta0) ; 
    q1.f.y = float(sinTheta0) ; 
    q1.f.z = float(cosTheta1) ; 
    q1.f.w = float(sinTheta1) ; 

    LOG(LEVEL) 
        << " startTheta_pi " << startTheta_pi
        << " deltaTheta_pi " << deltaTheta_pi
        << " theta0 " << theta0
        << " theta1 " << theta1
        << " cosTheta0Sign " << cosTheta0Sign 
        << " tan2Theta0 " << tan2Theta0
        << " cosTheta1Sign " << cosTheta1Sign
        << " tan2Theta1 " << tan2Theta1
        ;   


}

