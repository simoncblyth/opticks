// name=ssincosTest ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include "ssincos.h"
#include "SRng.hh"

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif


int main(int argc, char** argv)
{
    unsigned seed = 0 ; 

    SRng<double> rng(seed); 

    double sinPhi, cosPhi, u ; 

    for(unsigned i=0 ; i < 10 ; i++)  
    {
        u = rng();

        double phi = 2.*M_PIf*u ; 

        ssincos(phi,sinPhi,cosPhi);

        double dd = sinPhi*sinPhi + cosPhi*cosPhi ; 

        std::cout << u << " " << sinPhi << " " << cosPhi << " " << dd << std::endl ; 
    }
}

