#include "OpticksCSG.h"
#include "NThetaCut.hpp"

int main(int argc, char** argv)
{
    OpticksCSG_t type = CSG_LTHETACUT ; 

    double theta0_pi = 0. ; 
    double theta1_pi = 0.5 ; 

    nthetacut* n = make_thetacut( type, theta0_pi, theta1_pi ); 

    assert( n ); 

    return 0 ; 
}
