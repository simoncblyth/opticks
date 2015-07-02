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
 
#define RSAVE(seqhis, seqmat, p, s, slot)  \
{    \
    unsigned int shift = slot*4 ; \
    unsigned long long his = __ffs((s).flag) & 0xF ; \
    unsigned long long mat = __ffs((s).index.x) & 0xF ; \
    seqhis |= his << shift ; \
    seqmat |= mat << shift ; \
    unsigned int slot_offset =  (slot) < MAXREC  ? photon_id*MAXREC + (slot) : photon_id*MAXREC + MAXREC - 1 ;  \
    rsave((p), (s), record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
    (slot)++ ; \
}   \




/*

testing sequence formation with 
    /usr/local/env/numerics/thrustrap/bin/PhotonIndexTest


with seqhis = sflag   get expected values

print_vector  :                         histogram values 3 4 5 6 b c 
print_vector  :                         histogram counts 59493 453837 420 357 12861 85873  total: 612841


with seqhis = tmp  get expected values

print_vector  :                         histogram values 3 4 30 40 300 400 3000 4000 30000 40000 300000 400000 3000000 
                           4000000 30000000 40000000 300000000 400000000 500000000 600000000 b00000000 c00000000 


with seqhis = seqhis | tmp   get unexpected "f"  always appearing in most signficant 4 bits, 
                             skipping the 2nd seqset avoids the issue


   print_vector  :                         histogram values 3 5 31 51 61 c1 f1 361 3b1 3c1 551 561 5c1 651 661 6b1 6c1 c51 c61 cb1 cc1 f51 f61 fb1 fc1 3bb1 3bc1 3cb1 3cc1 5551 



*/



rtBuffer<float4>    genstep_buffer;
rtBuffer<float4>    photon_buffer;
rtBuffer<short4>    record_buffer;   // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long and unsigned long long are both 8 bytes, 64 bits 

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

    unsigned long long seqhis(0) ;
    unsigned long long seqmat(0) ;
    State s ;   
    Photon p ;  

    if(ghead.i.x < 0)   // 1st 4 bytes, is 1-based int index distinguishing cerenkov/scintillation
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) csdebug(cs);
#endif
        generate_cerenkov_photon(p, cs, rng );         
        s.flag = CERENKOV ;  
    }
    else
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset);
#ifdef DEBUG
        if(photon_id == 0) ssdebug(ss);
#endif
        generate_scintillation_photon(p, ss, rng );         
        s.flag = SCINTILLATION ;  
    }

    p.flags.u.y = photon_id ; 


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
            s.flag = MISS ;  // overwrite CERENKOV/SCINTILLATION for the no hitters
            s.index.x = 0 ;  // avoid unset m1 for no-hitters
            break ;
        }   
        // initial and CONTINUE-ing records

        p.flags.i.x = prd.boundary ;  

        // use boundary index at intersection point to do optical constant + material/surface property lookups 
        fill_state(s, prd.boundary, prd.sensor, p.wavelength );
        s.distance_to_boundary = prd.distance_to_boundary ; 
        s.surface_normal = prd.surface_normal ; 
        s.cos_theta = prd.cos_theta ; 

        p.flags.u.z = s.index.x ;   // material1 index 

        p.flags.u.w |= s.flag ; 


        if(photon_id == 0)
        {
           rtPrintf("bounce %d \n", bounce);
           rtPrintf("post  %10.3f %10.3f %10.3f %10.3f  % \n", p.position.x, p.position.y, p.position.z, p.time );
           rtPrintf("polw  %10.3f %10.3f %10.3f %10.3f  % \n", p.polarization.x, p.polarization.y, p.polarization.z, p.wavelength );
        } 


        RSAVE(seqhis, seqmat, p, s, slot) ;

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
    psave(p, photon_buffer, photon_offset ); 

    // building history sequence, bounce by bounce
    // s.flag is unexpected coming out 0xf 20% of time ? always here on the last placement ?
    // skipping the 2nd seqset :
    //    0) avoids the problem of trailing f 
    //    1) causes all the MISS to get zero, the initial seqhis value
    //

    RSAVE(seqhis, seqmat, p, s, slot) ;

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  

    rng_states[photon_id] = rng ;
}




RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






