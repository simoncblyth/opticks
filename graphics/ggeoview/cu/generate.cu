// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu

#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "define.h"
#include "PerRayData_propagate.h"

using namespace optix;

#include "quad.h"
#include "wavelength_lookup.h"
#include "photon.h"

#define GNUMQUAD 6
#include "cerenkovstep.h"
#include "scintillationstep.h"
#include "state.h"
#include "rayleigh.h"
#include "propagate.h"



// beyond MAXREC overwrite save into top slot
#define RSAVE(p, slot)  \
{    \
    unsigned int slot_offset =  slot < MAXREC  ? photon_id*MAXREC + (slot) : photon_id*MAXREC + MAXREC - 1 ;  \
    rsave((p), record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
    (slot)++ ; \
}   \



rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;
rtBuffer<short4>    record_buffer;   // 2 short4 take same space as 1 float4 quad


rtBuffer<curandState, 1> rng_states ;

rtDeclareVariable(float4,        center_extent, , );
rtDeclareVariable(float4,        time_domain  , , );
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

    // not combining State and PRD as assume minimal PRD advantage exceeds copying cost 

    State s ;   // perhaps rename to Boundary 

    PerRayData_propagate prd;
    prd.boundary = 0 ;
    prd.distance_to_boundary = -1.f ;

    Photon p ;  

    if(ghead.i.x < 0)   // 1st 4 bytes, is 1-based int index distinguishing cerenkov/scintillation
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) csdebug(cs);
#endif
        generate_cerenkov_photon(p, cs, rng );         
    }
    else
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) ssdebug(ss);
#endif
        generate_scintillation_photon(p, ss, rng );         
    }


    p.flags.u.y = photon_id ;   // no problem fitting uint  (1 << 32) - 1 = 4,294,967,295


    int slot = 0 ;
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

        RSAVE(p, slot) ;

#ifdef DEBUG
        if(photon_id == 0) dump_state(s);
#endif

        command = propagate_to_boundary( p, s, rng );
        if(command == BREAK)    break ; 
        if(command == CONTINUE) continue ; 

        if(s.surface.x > -1.f )      // x/y/z/w:detect/absorb/reflect_specular/reflect_diffuse
        {
            command = propagate_at_surface(p, s, rng);
            if(command == BREAK)    break ; 
            if(command == CONTINUE) continue ; 
        }
        else
        {
            propagate_at_boundary(p, s, rng);
        }

    }   // bounce < max_bounce

    psave(p, photon_buffer, photon_offset ); 

    RSAVE(p, slot) ;


    rng_states[photon_id] = rng ;
}




RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






