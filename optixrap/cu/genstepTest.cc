
#include "cudamock.h"

#include "quad.h"
#include "photon.h"
#include "Genstep_DsG4Scintillation_r4693.h"

int main(int argc, char** argv)
{
    float low = 0.f ; 
    float high = 10.f ; 

    curandState cs("/tmp/blyth/opticks/TRngBufTest_0.npy") ;   

    for(unsigned i=0 ; i < 10 ; i++)
    {
        float u = uniform( &cs, low, high ); 
        printf(" u : %10.3f \n", u ); 
    }

    for(unsigned i=0 ; i < 10 ; i++)
    {
        float3 sp = uniform_sphere( &cs ); 
        printf(" sp : (%10.3f, %10.3f, %10.3f)  \n", sp.x, sp.y, sp.z ); 
    }

    return 0 ; 
}
