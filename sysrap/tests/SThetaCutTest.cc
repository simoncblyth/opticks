#include "scuda.h"
#include "squad.h"
#include "SThetaCut.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    double startTheta_pi = 0.25 ; 
    double deltaTheta_pi = 0.50 ; 

    quad q0, q1 ; 
    SThetaCut::PrepareParam( q0, q1, startTheta_pi, deltaTheta_pi ); 


    return 0 ; 
}
