// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu

#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "define.h"
#include "PerRayData_propagate.h"

using namespace optix;

#include "quad.h"
#include "wavelength_lookup.h"

rtBuffer<uint4>                optical_buffer; 

#include "state.h"
#include "photon.h"

#define GNUMQUAD 6
#include "cerenkovstep.h"
#include "scintillationstep.h"

#include "rayleigh.h"
#include "propagate.h"


rtBuffer<float4>               genstep_buffer;
rtBuffer<float4>               photon_buffer;
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 

rtBuffer<unsigned char>        phosel_buffer; 
rtBuffer<unsigned char>        recsel_buffer; 

rtBuffer<curandState, 1>       rng_states ;

rtDeclareVariable(float4,        center_extent, , );
rtDeclareVariable(float4,        time_domain  , , );
rtDeclareVariable(float,         propagate_epsilon, , );
rtDeclareVariable(unsigned int,  propagate_ray_type, , );
rtDeclareVariable(unsigned int,  bounce_max, , );
rtDeclareVariable(unsigned int,  record_max, , );
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

// beyond MAXREC overwrite save into top slot
 
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

RT_PROGRAM void trivial()
{
    unsigned long long photon_id = launch_index.x ;  
    if(photon_id == 0)
    {
       rtPrintf("trivial\n");
    } 
}

RT_PROGRAM void generate()
{
    union quad phead ;
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    unsigned int MAXREC = record_max ; 
    if(photon_id == 0)
    {
       rtPrintf("generate\n");
    } 
 
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
        // * each turn of the loop only sets a single history bit  
        // * bit position is used rather than the full mask in the photon record
        //   to save bits  
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






