#include "S4MTRandGaussQ.h"
#include "NP.hh"

int main()
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

    a->save("$FOLD/a.npy"); 

    return 0 ; 
}
