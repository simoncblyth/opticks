#include "OpticksCSG.h"
#include "NPhiCut.hpp"

int main(int argc, char** argv)
{
    OpticksCSG_t type = CSG_PHICUT ; 

    double startPhi_pi = 0. ; 
    double deltaPhi_pi = 0.5 ; 

    nphicut* n = make_phicut( type, startPhi_pi, deltaPhi_pi ); 

    assert( n ); 

    return 0 ; 
}
