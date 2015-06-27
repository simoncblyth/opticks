// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu


#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "define.h"
#include "PerRayData_propagate.h"

using namespace optix;

#include "quad.h"
#include "wavelength_lookup.h"
#include "state.h"
#include "photon.h"

#define GNUMQUAD 6
#include "cerenkovstep.h"
#include "scintillationstep.h"

#include "rayleigh.h"
#include "propagate.h"


// beyond MAXREC overwrite save into top slot
//    if(photon_id == 0) dump_state((s));
 
#define RSAVE(p, s, slot)  \
{    \
    unsigned int slot_offset =  (slot) < MAXREC  ? photon_id*MAXREC + (slot) : photon_id*MAXREC + MAXREC - 1 ;  \
    rsave((p), (s), record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
    (slot)++ ; \
}   \



rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;
rtBuffer<short4>    record_buffer;   // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   history_buffer;   // unsigned long and unsigned long long are both 8 bytes, 64 bits 

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
    unsigned long long seqhis = 0 ;
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
 
    phead.f = photon_buffer[photon_offset+0] ;

    union quad ghead ; 
    unsigned int genstep_id = phead.u.x ; // first 4 bytes seeded with genstep_id
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

    curandState rng = rng_states[photon_id];

    // not combining State and PRD as assume minimal PRD advantage exceeds copying cost 

    State s ;   // perhaps rename to Step (as in propagation, not generation) 

    Photon p ;  

    if(ghead.i.x < 0)   // 1st 4 bytes, is 1-based int index distinguishing cerenkov/scintillation
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) csdebug(cs);
#endif
        generate_cerenkov_photon(p, s, cs, rng );         
    }
    else
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) ssdebug(ss);
#endif
        generate_scintillation_photon(p, s, ss, rng );         
    }

    p.flags.u.y = photon_id ;   // no problem fitting uint  (1 << 32) - 1 = 4,294,967,295


    int slot = 0 ;
    int bounce = 0 ; 
    int command = START ; 

    PerRayData_propagate prd ;

    while( bounce < bounce_max )
    {
        bounce++;   // increment at head, not tail, as CONTINUE skips the tail
        prd.boundary = 0 ;
        prd.sensor = 0 ;
        prd.boundary = 0 ;
        prd.distance_to_boundary = -1.f ;

        rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );
        // see material1_propagate.cu:closest_hit_propagate

        if(prd.boundary == 0)
        {
            //p.flags.i.w |= NO_HIT;
            s.flag = MISS ;  // overwrite CERENKOV/SCINTILLATION for the no hitters
            break ;
        }   

        p.flags.i.x = prd.boundary ;  

        // use boundary index at intersection point to do optical constant + material/surface property lookups 
        fill_state(s, prd.boundary, prd.sensor, p.wavelength );
        s.distance_to_boundary = prd.distance_to_boundary ; 
        s.surface_normal = prd.surface_normal ; 
        s.cos_theta = prd.cos_theta ; 

        p.flags.u.z = s.index.x ;   // material1 index 

        // initial and CONTINUE-ing records
        p.flags.u.w |= s.flag ; 


        if(photon_id == 0)
        {
           rtPrintf("bounce %d \n", bounce);
           rtPrintf("post  %10.3f %10.3f %10.3f %10.3f  % \n", p.position.x, p.position.y, p.position.z, p.time );
           rtPrintf("polw  %10.3f %10.3f %10.3f %10.3f  % \n", p.polarization.x, p.polarization.y, p.polarization.z, p.wavelength );
        } 


        seqhis |= s.flag << (bounce - 1)*4 ;  // building history sequence, bounce by bounce
        RSAVE(p, s, slot) ;

        // Where best to record the propagation ? 
        // =======================================
        //
        // The above code: 
        // ~~~~~~~~~~~~~~~~   
        //                             /\
        //      *--> . . . . . . m1  ./* \ - m2 - - - -
        //     p                     / su \
        //                          /______\
        //
        // * finds intersected triangle along ray from p.position along p.direction
        // * uses triangle primIdx to find boundaryIndex 
        // * use boundaryIndex and p.wavelength to look up surface/material properties 
        //   including s.index.x/y/z  m1/m2/su indices 
        // 
        // * **NB ABOVE CODE DOES NOT change p.direction/p.position/p.time** 
        //   they are still at their generated OR last changed positions 
        //   from CONTINUE-ers from the below propagation code 
        //
        //  In summary the above code is looking ahead to see where to go next
        //  while the photon remains with exitant position and direction
        //  and flags from the last "step"
        //
        // The below code:
        // ~~~~~~~~~~~~~~~~
        //
        // * changes p.position p.time p.direction p.polarization ... 
        //   then CONTINUEs back to the above to find the next intersection
        //   or BREAKs
        //
        //
        //            BULK_SCATTER/BULK_REEMIT
        //                 ^
        //                 |           /\
        //      *--> . # . *. . . m1  /* \ - m2 - - - -
        //     p       .             / su \
        //          BULK_ABSORB     /______\
        //
        //
        //
        // TODO:
        //
        // * arrange for each record to only set a single history bit  
        // * use bit position rather than the full mask in the photon record 
        //
        //   0:31 fits in 5 bits 
        //   0:15 fits in 4 bits
        //
        // * control the or-ing of that flag into photon history at this level
        // * integrate surface optical props: finish=polished/ground
        //

        command = propagate_to_boundary( p, s, rng );
        if(command == BREAK)    break ;           // BULK_ABSORB
        if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
        //
        // PASS : survivors will go on to pick up one of the below flags, 
        //        so no need for "BULK_SURVIVE"
       
        if(s.surface.x > -1.f )  // x/y/z/w:detect/absorb/reflect_specular/reflect_diffuse
        {
            command = propagate_at_surface(p, s, rng);
            if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
            if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
        }
        else
        {
            propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
            // tacit CONTINUE
        }

    }   // bounce < max_bounce


    // breakers and maxers saved here
    p.flags.u.w |= s.flag ; 
    seqhis |= s.flag << (bounce - 1)*4 ;  // building history sequence, bounce by bounce
    psave(p, photon_buffer, photon_offset ); 
    RSAVE(p, s, slot) ;

    history_buffer[photon_id] = seqhis ; 
    rng_states[photon_id] = rng ;
}




RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






