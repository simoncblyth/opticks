#include "S4MTRandGaussQ.h"
#include "NP.hh"
#include "njuffa_erfcinvf.h"

void test_transformQuick()
{
    int N = 1000 ; 
    NP* a = NP::Make<double>( N, 2 ); 
    double* aa = a->values<double>() ;

    for(int i=0 ; i < N ; i++)
    {
        double r = double(i)/double(N-1)  ;   
        double v = S4MTRandGaussQ::transformQuick(r) ; 
        aa[i*2+0] = r ; 
        aa[i*2+1] = v ; 
    }
    a->save("$FOLD/test_transformQuick.npy"); 
}

void test_transformQuick_vs_njuffa_erfcinvf()
{
    int N = 1000 ;

    int ni = N ; 
    int nj = 3 ; 
 
    NP* a = NP::Make<double>( ni, nj ); 
    double* aa = a->values<double>() ;

    for(int i=0 ; i < ni ; i++)
    {
        double r = double(i)/double(N-1)  ;   
        aa[i*nj+0] = r ; 
        aa[i*nj+1] = S4MTRandGaussQ::transformQuick(r)  ; 
        aa[i*nj+2] = -sqrtf(2.f)*njuffa_erfcinvf(r*2.f) ; 
        // argument to erfcinvf needs to be in range 0->2 
        // then have to scale to match S4MTRandGaussQ::transformQuick
    }
    a->save("$FOLD/test_transformQuick_vs_njuffa_erfcinvf.npy"); 
}


void test_shoot()
{
    int N = 1000000 ; 
    NP* a = NP::Make<double>( N ); 
    double* aa = a->values<double>() ;

    double mean = 0.0 ; 
    double stdDev = 0.1 ; 

    for(int i=0 ; i < N ; i++) aa[i] = S4MTRandGaussQ::shoot(mean, stdDev) ; 
    a->save("$FOLD/test_shoot.npy"); 
}


int main()
{
    //test_transformQuick(); 
    //test_shoot(); 

    test_transformQuick_vs_njuffa_erfcinvf(); 

    return 0 ; 
}
