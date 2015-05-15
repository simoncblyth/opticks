
#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "PerRayData_propagate.h"

using namespace optix;

#include "uif.h"
#include "quad.h"
#include "wavelength_lookup.h"
#include "cerenkovstep.h"

rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<curandState, 1> rng_states ;


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

    PerRayData_propagate prd;
    prd.depth = 0 ;
    prd.rng = rng_states[photon_id];

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
            cscheck(cs);
            reemission_check();
            //wavelength_check();
        }
    }
    else
    {

    }

    // arbitrarily setting photon positions to genstep position with offset : to check visualization of generated photons
    //float scale = 100.f * (photon_id % 100) ;  
    float u = curand_uniform(&prd.rng);
    float nm = reemission_lookup(u);
    float scale = 10000.f * u ;
    photon_buffer[launch_index.x] = make_float4( gs1.x + gs2.x*scale, gs1.y + gs2.y*scale, gs1.z + gs2.z*scale, nm );
 
    rng_states[photon_id] = prd.rng ;
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






