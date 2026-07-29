/**
~/o/sysrap/tests/scurand_test.sh

**/


#include <cstdio>

#include "srngcpu.h"
using RNG = srngcpu ;

#include "scuda.h"
#include "scurand.h"
#include "ssys.h"


struct scurand_test
{
    static int uniform();
    static int shoot_exponential();
    static int main();
};


inline int scurand_test::uniform()
{
    printf("//scurand_test::uniform\n");
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
    return 0;
}

inline int scurand_test::shoot_exponential()
{
    RNG rng ;
    printf("//scurand_test::shoot_exponential\n");

    for(int i=0 ; i < 20 ; i++)
    {
        float uf = scurand<float>::shoot_exponential(&rng);
        printf("// %2d uf %10.4f \n", i, uf );
    }
    for(int i=0 ; i < 20 ; i++)
    {
        double ud = scurand<double>::shoot_exponential(&rng) ;
        printf("// %2d ud %10.4f \n", i, ud );
    }
    return 0;
}

inline int scurand_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"uniform")) rc += uniform();
    if(ALL||0==strcmp(TEST,"shoot_exponential")) rc += shoot_exponential();
    return rc ;
}

int main()
{
    return scurand_test::main();
}
