/**

::

        .                            phi0=0.50

                                      Y
                                  .   -     
                              .       |         
                                      |    pp quadrant cutaway with pacmanpp
                           .          |           
                                      ^  (0,1,0) normal
                          .          /|\
                                      |             
        -X  0 -----------1------------2--->---------3---- phi1=2.0 --------- +X
                                      |  (1,0,0) dir 
         -150          -100           0            100
                                      |  
                           .          |          .
                                      |      
                              .       |       .
                                  .   -   .

**/

#include <vector>

#include "scuda.h"
#include "squad.h"
#include "SPhiCut.hh"
#include "SMath.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    double phiStart_pi = 0.50 ; 
    double phiDelta_pi = 1.50 ; 
    double phi0_pi = phiStart_pi ; 
    double phi1_pi = phiStart_pi + phiDelta_pi ; 

    double cosPhi0 = std::cos(phi0_pi*M_PI ); 
    double sinPhi0 = std::sin(phi0_pi*M_PI ); 
    double cosPhi1 = std::cos(phi1_pi*M_PI ); 
    double sinPhi1 = std::sin(phi1_pi*M_PI ); 

    double _cosPhi0 = SMath::cos_pi(phi0_pi); 
    double _sinPhi0 = SMath::sin_pi(phi0_pi); 
    double _cosPhi1 = SMath::cos_pi(phi1_pi); 
    double _sinPhi1 = SMath::sin_pi(phi1_pi); 

    std::vector<std::pair<std::string, double>> pairs = 
    { 
        {"phiStart_pi", phiStart_pi },
        {"phiDelta_pi", phiDelta_pi },
        {"phi0_pi",     phi0_pi },
        {"phi1_pi",     phi1_pi },
        {"M_PIf",       M_PIf},
        {"M_PI",        M_PI},
        {"cosPhi0",     cosPhi0},
        {"sinPhi0",     sinPhi0},
        {"cosPhi1",     cosPhi1},
        {"sinPhi1",     sinPhi1},
        {"_cosPhi0",    _cosPhi0},
        {"_sinPhi0",    _sinPhi0},
        {"_cosPhi1",    _cosPhi1},
        {"_sinPhi1",    _sinPhi1},
    };  

    std::cout << SMath::Format(pairs); 

    quad q0 ; 
    SPhiCut::PrepareParam( q0, phiStart_pi, phiDelta_pi ); 

    LOG(info) << q0.f ; 

    return 0 ; 
}
