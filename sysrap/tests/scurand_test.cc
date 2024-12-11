/**
~/o/sysrap/tests/scurand_test.sh 

**/


#include <cstdio>

#include "srngcpu.h"
using RNG = srngcpu ; 

#include "scurand.h"

int main()
{
    RNG rng ;   

    for(int i=0 ; i < 20 ; i++)
    {
        float uf = scurand<float>::uniform(&rng) ;
        printf("// %2d uf %10.4f \n", i, uf ); 
    }

    for(int i=0 ; i < 20 ; i++)
    {
        double ud = scurand<double>::uniform(&rng) ;
        printf("// %2d ud %10.4f \n", i, ud ); 
    }

    return 0 ; 
}
