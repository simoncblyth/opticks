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
#include "state.h"
#include "rayleigh.h"
#include "propagate.h"


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
    prd.boundary = 0 ;
    prd.distance_to_boundary = -1.f ;

    // combine State and PRD ?
    //    * currently no, due to assumption that a minimal PRD
    //      is worth the cost of shuffling some results from PRD to State

    State s ; 

    Photon p ;  
    pinit(p);


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

        // pol and dir distrib for scintillation are flat, so 
        // test rayleigh here 
        // rayleigh_scatter(p, rng);

    }


    int bounce = 0 ; 
    int command ; 

    while( bounce < bounce_max )
    {
        bounce++;

        optix::Ray ray = optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX);

        rtTrace(top_object, ray, prd);  // see material1_propagate.cu:closest_hit_propagate

        if(prd.boundary == 0)
        {
            p.flags.i.w |= NO_HIT;
            break ;
        }     

        p.flags.i.x = prd.boundary ;  

        fill_state(s, prd.boundary, p.wavelength );

        s.distance_to_boundary = prd.distance_to_boundary ; 
        s.surface_normal = prd.surface_normal ; 
        s.cos_theta = prd.cos_theta ; 

        if(photon_id == 0)
        {
            dump_state(s);
        }


        command = propagate_to_boundary( p, s, rng );
        if(command == BREAK)    break ; 
        if(command == CONTINUE) continue ; 


        /*
        // DEBUG: over-writing third quad : vpol as expedient for access from ../gl/pos/vert.glsl
        p.position += prd.distance_to_boundary*p.direction ; 
        p.polarization.x = prd.cos_theta ; 
        p.polarization.y = prd.distance_to_boundary ; 
        p.polarization.z = 0.f ; 
        p.weight = 0.f ;   
        */


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






