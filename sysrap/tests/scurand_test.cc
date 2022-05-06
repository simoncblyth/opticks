// name=scurand_test ; gcc $name.cc -g -I.. -DMOCK_CURAND -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cstdio>
#include "scurand.h"

int main()
{
    curandStateXORWOW rng(1u) ;   

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
