#include <optix_world.h>
using namespace optix;

rtBuffer<float3, 1>  cgb_buffer;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );


RT_PROGRAM void cgb()
{
    float frac = float(launch_index)/float(launch_dim) ; 
    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*frac,&sinPhi,&cosPhi);
    rtPrintf( "cgb.cu launch dim %d index %d frac %10.4f s %10.4f c %10.4f \n", launch_dim, launch_index, frac, sinPhi, cosPhi);
    cgb_buffer[launch_index] = make_float3( sinPhi,  cosPhi,  0.0f) ;
}




