#include "OPTICKS_LOG.hh"
#include "OpticksCSG.h"
#include "NThetaCut.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    OpticksCSG_t type = CSG_THETACUT ; 

    double thetaStart_pi = 0.25 ; 
    double thetaDelta_pi = 0.50 ; 
    nthetacut* n = nthetacut::make( type, thetaStart_pi, thetaDelta_pi ); 

    assert( n ); 

    return 0 ; 
}
