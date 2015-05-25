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

rtDeclareVariable(float,         propagate_epsilon, , );
rtDeclareVariable(unsigned int,  propagate_ray_type, , );
rtDeclareVariable(unsigned int,  bounce_max, , );
rtDeclareVariable(rtObject,      top_object, , );

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

    curandState rng = rng_states[photon_id];

    PerRayData_propagate prd;
    prd.boundary = -1 ;
    prd.distance_to_boundary = -1.f ;

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
        generate_cerenkov_photon(p, cs, rng );         
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
        generate_scintillation_photon(p, ss, rng );         
    }


    int bounce = 0 ; 
    while( bounce < bounce_max )
    {
        bounce++;

        optix::Ray ray = optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX);

        rtTrace(top_object, ray, prd);       // see material1_propagate.cu:closest_hit_propagate

        // what happens with photons that miss ? 
        p.position += prd.distance_to_boundary*p.direction ; 

        // over-writing third quad : vpol as expedient for access from ../gl/pos/vert.glsl
        p.polarization.x = prd.cos_theta ; 
        p.polarization.y = prd.distance_to_boundary ; 
        p.polarization.z = 0.f ; 
        p.weight = 0.f ;   

        // fourth quad : is accessible in shaders as ivec4 so keep int here 
        unsigned int boundary_code = prd.boundary + 1 ;   // 1-based for cos_theta signing, 0 means miss
        p.flags.i.x = prd.cos_theta < 0.f ? -boundary_code : boundary_code ;
        p.flags.i.y = 0 ;
        p.flags.i.z = 0 ; 
        p.flags.i.w = 0 ; 

        float4 imat = wavelength_lookup( p.wavelength,  prd.boundary*6 + 0 );
        float4 omat = wavelength_lookup( p.wavelength,  prd.boundary*6 + 1 );

        if(photon_id == 0)
        {
            rtPrintf(" prd t/ct/boundary %10.4f %10.4f %d \n", prd.distance_to_boundary, prd.cos_theta, prd.boundary );
            rtPrintf(" imat %10.4f %10.4f %10.4f %10.4f \n", imat.x, imat.y, imat.z, imat.w );
            rtPrintf(" omat %10.4f %10.4f %10.4f %10.4f \n", omat.x, omat.y, omat.z, omat.w );
        }

    }  // bounce < max_bounce

    psave(p, photon_buffer, photon_offset ); 
    rng_states[photon_id] = rng ;
}



/*

In [1]: a = oxc_(1)

In [4]: a[:,3,0].view(np.int32)
Out[4]: array([11, 11, 11, ..., -1, -1, -1], dtype=int32)

In [5]: np.unique(a[:,3,0].view(np.int32))
Out[5]: array([-1, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 24, 31, 32, 49, 50, 52], dtype=int32)

plt.hist(a[:,3,1], bins=100, log=True)

*/



RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






