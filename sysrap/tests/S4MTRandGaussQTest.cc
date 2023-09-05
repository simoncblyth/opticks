#include "S4MTRandGaussQ.h"
#include "NP.hh"

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

void test_shoot()
{
    int N = 1000000 ; 
    NP* a = NP::Make<double>( N ); 
    double* aa = a->values<double>() ;

    double mean = 5. ; 
    double stdDev = 0.01 ; 

    for(int i=0 ; i < N ; i++) aa[i] = S4MTRandGaussQ::shoot(mean, stdDev) ; 
    a->save("$FOLD/test_shoot.npy"); 
}


int main()
{
    //test_transformQuick(); 
    test_shoot(); 
    return 0 ; 
}
