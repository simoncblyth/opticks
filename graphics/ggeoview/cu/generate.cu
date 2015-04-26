#include <optix_world.h>
#include "uif.h"

using namespace optix;

rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


RT_PROGRAM void generate()
{
    unsigned long long iph = launch_index.x ;  
    float4 ph = photon_buffer[iph] ;

    uif_t uif     ; 
    uif.f = ph.x  ;   
    unsigned int igs = uif.u ; 

    float4 gs0 = genstep_buffer[igs*6+0];
    float4 gs1 = genstep_buffer[igs*6+1];
    float4 gs2 = genstep_buffer[igs*6+2];

    //rtPrintf("generate.cu::generate iph %7d igs %7d  "
    //         "gs2.xyzw %10.3f %10.3f %10.3f %10.3f \n", iph,  igs, gs2.x, gs2.y, gs2.z, gs2.w );
    //    "gs1.xyzw %10.3f %10.3f %10.3f %10.3f \n", iph,  igs, gs1.x, gs1.y, gs1.z, gs1.w );
    //

    // arbitrarily setting photon positions to genstep position 
    // with offset : to check visualization of generated photons
    //
    float scale = 100.f * (iph % 100) ;  
    photon_buffer[launch_index.x] = make_float4( gs1.x + gs2.x*scale, gs1.y + gs2.y*scale, gs1.z + gs2.z*scale, 1.f );
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






