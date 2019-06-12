#include "OPTICKS_LOG.hh"
#include "cfloat4x4.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    cfloat4x4 pho ; 

    pho.q0 = make_float4( 0.5f,   1.5f,  2.5f,  3.5f );     
    pho.q1 = make_float4( 10.5f, 11.5f, 12.5f, 13.5f );     
    pho.q2 = make_float4( 20.5f, 21.5f, 22.5f, 23.5f );     

    tquad q3 ; 
    q3.u = make_uint4(    42, 43, 44, 45 );  
    pho.q3 = q3.f ; 

    LOG(info) << pho ; 

    return 0 ; 
}


