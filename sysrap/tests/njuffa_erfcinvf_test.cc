#include "njuffa_erfcinvf.h"
#include "NP.hh"

int main()
{
    const int ni = 1000 ; 
    NP* a = NP::Make<float>(ni, 2); 
    float* aa = a->values<float>(); 

    float SQRT2 = sqrtf(2.f) ; 

    for(int i=0 ; i < ni ; i++ )
    {
        float u = 2.f*float(i)/float(ni-1) ; 
        float v = -SQRT2*njuffa_erfcinvf(u);          
        aa[2*i+0] = u ; 
        aa[2*i+1] = v ; 
    } 
    a->save("$FOLD/njuffa_erfcinvf_test.npy"); 

    return 0 ; 
}
