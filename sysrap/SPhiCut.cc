#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "SMath.hh"
#include "SPhiCut.hh"

const plog::Severity SPhiCut::LEVEL = PLOG::EnvLevel("SPhiCut", "DEBUG" ); 

/**
SPhiCut::PrepareParam 
------------------------

This is used by

* npy/NPhiCut.cpp
* CSG/CSGNode.cc


The input parameters: startPhi, deltaPhi configure two azimuthal angles in range [0., 2.] (units of pi)

* startPhi
* startPhi+deltaPhi

A do nothing phicut shape would have have startPhi=0. deltaPhi=2. 


               Y

              0.5 
               :   /
               :  /
               : /
               :/ 
     1.0- - -  O - - -  0.0    X
               :\
               : \
               :  \
               :   \

              1.5 


+---------------------------+----------------------+---------------------------------+-------------------------------+
| (startPhi, deltaPhi)      | (phi0, phi1)         |  Notes                          |  Cutaway                      |
+===========================+======================+=================================+===============================+
|  (0.0, 2.0)               |  (0.0, 2.0)          |  Full range of phi              |                               |
+---------------------------+----------------------+---------------------------------+-------------------------------+

**/




void SPhiCut::PrepareParam( quad& q0, double startPhi_pi, double deltaPhi_pi )
{
    double phi0_pi = startPhi_pi ; 
    double phi1_pi = startPhi_pi + deltaPhi_pi ;

    double cosPhi0 = SMath::cos_pi(phi0_pi) ;
    double sinPhi0 = SMath::sin_pi(phi0_pi) ;
    double cosPhi1 = SMath::cos_pi(phi1_pi) ;
    double sinPhi1 = SMath::sin_pi(phi1_pi) ;

    q0.f.x = float(cosPhi0); 
    q0.f.y = float(sinPhi0); 
    q0.f.z = float(cosPhi1); 
    q0.f.w = float(sinPhi1); 

    LOG(LEVEL) 
        << " startPhi_pi " << startPhi_pi
        << " deltaPhi_pi " << deltaPhi_pi
        << " phi0_pi " << phi0_pi
        << " phi1_pi " << phi1_pi
        << " cosPhi0 " << cosPhi0
        << " sinPhi0 " << sinPhi0 
        << " cosPhi1 " << cosPhi1
        << " sinPhi1 " << sinPhi1
        ;   


}

