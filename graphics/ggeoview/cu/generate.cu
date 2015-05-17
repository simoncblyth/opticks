// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu

#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "PerRayData_propagate.h"

using namespace optix;

//#include "uif.h"
#include "quad.h"
#include "photon.h"
#include "wavelength_lookup.h"

#define GNUMQUAD 6
#include "cerenkovstep.h"
#include "scintillationstep.h"

rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;
rtBuffer<curandState, 1> rng_states ;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

RT_PROGRAM void generate()
{
    union quad phead ;
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    phead.f = photon_buffer[photon_offset+0] ;

    union quad ghead ; 
    unsigned int genstep_id = phead.u.x ; // first 4 bytes seeded with genstep_id
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

    PerRayData_propagate prd;
    prd.depth = 0 ;
    prd.rng = rng_states[photon_id];

    Photon p ;  

    if(ghead.i.x < 0)   // 1st 4 bytes, is 1-based int index distinguishing cerenkov/scintillation
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset);

        if(photon_id == 0)
        {
            csdump(cs);
            cscheck(cs);
            //wavelength_check();
        }
        generate_cerenkov_photon(p, cs, prd.rng );         
    }
    else
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset);

        if(photon_id == 0)
        {
            ssdump(ss);
            reemission_check();
        }
        generate_scintillation_photon(p, ss, prd.rng );         
    }

    // TODO: fix shader to avoid having to do this kludge to see smth with OpenGL viz
    //p.position += p.direction*1000.f ; 

    psave(p, photon_buffer, photon_offset ); 
    rng_states[photon_id] = prd.rng ;
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






