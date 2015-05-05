#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
using namespace optix;

#include "uif.h"
#include "quad.h"
#include "cerenkovstep.h"
#include "wavelength_lookup.h"

rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu

RT_PROGRAM void generate()
{
    unsigned long long photon_id = launch_index.x ;  
    float4 ph = photon_buffer[photon_id] ;

    // first 4 bytes of initiating empty photon_buffer contains genstep_id
    uif_t uif     ; 
    uif.f = ph.x  ;   
    unsigned int genstep_id = uif.u ; 

    float4 gs0 = genstep_buffer[genstep_id*6+0];
    float4 gs1 = genstep_buffer[genstep_id*6+1];
    float4 gs2 = genstep_buffer[genstep_id*6+2];


    // first 4 bytes of genstep entry distinguishes 
    // cerenkov and scintillation by the sign of a 1-based index 
    union quad genstep_head ; 
    genstep_head.f = genstep_buffer[genstep_id*6+0]; 
    int genstep_head_id = genstep_head.i.x ; 

    if(genstep_head_id < 0)
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_id);
        csinit(cs);

        if(photon_id == 0)
        {
            csdump(cs);
            wavelength_check();
        }
    }
    else
    {

    }


    //rtPrintf("generate.cu::generate iph %7d igs %7d  "
    //         "gs2.xyzw %10.3f %10.3f %10.3f %10.3f \n", iph,  igs, gs2.x, gs2.y, gs2.z, gs2.w );
    //    "gs1.xyzw %10.3f %10.3f %10.3f %10.3f \n", iph,  igs, gs1.x, gs1.y, gs1.z, gs1.w );
    //

    // arbitrarily setting photon positions to genstep position 
    // with offset : to check visualization of generated photons
    //
    float scale = 100.f * (photon_id % 100) ;  
    photon_buffer[launch_index.x] = make_float4( gs1.x + gs2.x*scale, gs1.y + gs2.y*scale, gs1.z + gs2.z*scale, 1.f );
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






