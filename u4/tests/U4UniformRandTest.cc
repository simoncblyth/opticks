#include <cassert>

#include "U4UniformRand.h"
NP* U4UniformRand::UU = nullptr ; 

int main(int argc, char** argv)
{
    const int N = 1000 ; 
    NP* u = U4UniformRand::Get(N) ; 
    std::cout << u->repr<double>() << std::endl ; 

    U4UniformRand::UU = u ; 

    const double* uu = u->cvalues<double>(); 
    for(int i=0 ; i < N ; i++)
    {
        int idx0 = U4UniformRand::Find(uu[i], u) ; 
        int idx1 = U4UniformRand::Find(uu[i]) ; 

        std::cout 
            << " i " << std::setw(5) << i 
            << " Desc0 " << U4UniformRand::Desc(uu[i], u ) 
            << " Desc1 " << U4UniformRand::Desc(uu[i] ) 
            << " idx0 " << idx0 
            << " idx1 " << idx1 
            << std::endl 
            ;
        assert( idx0 == idx1 );  
        assert( idx0 == i );  
    }
    return 0 ; 
}
