// name=s_mock_curand_test  ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <cstdio>
#include "s_mock_curand.h"


void test_mock_curand_0(curandState_t& rng)
{
    for(int i=0 ; i < 10 ; i++) printf("//test_mock_curand_0 i %2d u %10.6f \n", i, curand_uniform(&rng) ); 
}

void test_mock_curand_1(curandStateXORWOW& rng)
{
    for(int i=0 ; i < 10 ; i++) printf("//test_mock_curand_1 i %2d u %10.6f \n", i, curand_uniform(&rng) ); 
}

void test_mock_curand_0()
{
    curandState_t rng(1u) ;   
    test_mock_curand_1(rng); 
}

void test_mock_curand_1()
{
    curandStateXORWOW rng(1u) ;   
    test_mock_curand_0(rng); 
}

int main(int argc, char** argv)
{
    test_mock_curand_0();
    test_mock_curand_1();
    return 0;
}


