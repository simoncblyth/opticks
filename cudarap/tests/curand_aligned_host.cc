/*

See :doc:`notes/issues/random_alignment`


https://devtalk.nvidia.com/default/topic/498171/how-to-get-same-output-by-curand-in-cpu-and-gpu/

For CURAND, we use the seed to pick a random spot to start, then cut the long
sequence into 4096 chunks each spaced 2^67 positions apart. The host API lets
you grab blocks of results from this shuffled sequence. If you request 8192
results, you will get the first result from each of the 4096 chunks, then the
second result from each of the 4096 chunks.

For the device API using curand_init(), you explicitly give the subsequence
number and manage the threads yourself. 

If you want to exactly match the results from the host API you need 
to launch 4096 total threads, then have each
one call curand_init() with the same seed and subsequence numbers from 0 to
4095. Then you need to store the results in a coalesced strided manner; thread
0 goes first with one value, then next in memory is thread 1 with one value,
then thread 2, etc.. 

The reason you are seeing the number 8192 is because you are generating double
precision values. Each double result uses 2 32-bit results.

*/
#include <cassert>
#include <cstdio>
#include <iostream>
#include <curand.h>


int main(int argc, char** argv)
{
    printf("%s\n", argv[0]) ;
    static const unsigned NI = 100000 ;  // beyond 4096 get wrap-back   
    static const unsigned NJ = 16 ;

    unsigned i0 = argc > 1 ? atoi(argv[1]) : 0 ; 
    unsigned i1 = argc > 2 ? atoi(argv[2]) : i0 + 1 ; 

    assert( i1 > i0 ); 
    assert( i0 < NI ); 
    assert( i1 <= NI ); 
   
    float data[NJ*NI] ;

    unsigned long long _seed = 0ull ; 

    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, _seed );

    printf("generate NJ %u clumps of NI %u : ", NJ, NI );
    for(unsigned j=0 ; j < NJ ; j++)
    { 
        printf(" %u ", j ); 
        curandSetGeneratorOffset(gen, j*4096 ); 
        curandGenerateUniform(gen, data + j*NI, NI );
    } 
    printf("\n"); 


    printf("dump i0:%u i1:%u \n", i0, i1) ;

    for( unsigned i=i0 ; i < i1 ; i++) 
    {
        printf("i:%u \n", i ); 
        for( unsigned j=0 ; j < NJ ; j++)
        { 
            printf("%lf ",*(data + j*NI + i)  );
            if( j % 4 == 3 ) printf("\n") ; 
        } 
    }

    curandDestroyGenerator(gen);
    return 0;
}
