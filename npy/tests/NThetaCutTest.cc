#include "OPTICKS_LOG.hh"
#include "OpticksCSG.h"
#include "NThetaCut.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    double thetaStart_pi = 0.25 ; 
    double thetaDelta_pi = 0.50 ; 
    nthetacut* n = nthetacut::make( thetaStart_pi, thetaDelta_pi ); 

    assert( n ); 

    return 0 ; 
}
