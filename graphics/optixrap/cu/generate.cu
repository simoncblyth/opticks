// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/generate.cu


#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "define.h"
#define DEBUG 1 
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
#include "torchstep.h"

#include "rayleigh.h"
#include "propagate.h"

rtBuffer<float4>               genstep_buffer;
rtBuffer<float4>               photon_buffer;
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 

//#define AUX 1
#ifdef AUX
rtBuffer<short4>                aux_buffer ; 
#endif


rtBuffer<curandState, 1>       rng_states ;

rtDeclareVariable(float4,        center_extent, , );
rtDeclareVariable(float4,        time_domain  , , );
rtDeclareVariable(uint4,         debug_control , , );
rtDeclareVariable(float,         propagate_epsilon, , );
rtDeclareVariable(unsigned int,  propagate_ray_type, , );
rtDeclareVariable(unsigned int,  bounce_max, , );
rtDeclareVariable(unsigned int,  record_max, , );
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

// beyond MAXREC overwrite save into top slot
 
#define RSAVE(seqhis, seqmat, p, s, slot, slot_offset)  \
{    \
    unsigned int shift = slot*4 ; \
    unsigned long long his = __ffs((s).flag) & 0xF ; \
    unsigned long long mat = (s).index.x < 0xF ? (s).index.x : 0xF ; \
    seqhis |= his << shift ; \
    seqmat |= mat << shift ; \
    rsave((p), (s), record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
}   \


// stomps on su (surface index) for 1st slot : inserting the genstep m1 
#define ASAVE(p, s, slot, slot_offset, MaterialIndex) \
{   \
   aux_buffer[slot_offset] = make_short4( \
             s.index.x, \
             s.index.y, \
             slot == 0 ? optical_buffer[MaterialIndex].x : s.index.z, \
             p.flags.i.x );  \
}  \
        



RT_PROGRAM void trivial()
{
   wavelength_dump(24, 10);    
   wavelength_dump(42, 10);    
   wavelength_dump(48, 10);    
   wavelength_dump(1048, 10);    

}



RT_PROGRAM void generate()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    unsigned int MAXREC = record_max ; 

    // first 4 bytes of photon_buffer photon records is seeded with genstep_id 
    // this seeding is done by App::seedPhotonsFromGensteps

    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;
    unsigned int genstep_id = phead.u.x ; 
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

#ifdef DEBUG
    bool dbg = photon_id == debug_control.x ;  
    if(dbg)
    {
       rtPrintf("generate debug photon_id %d genstep_id %d ghead.i.x %d \n", photon_id, genstep_id, ghead.i.x );
    } 
#endif 


    curandState rng = rng_states[photon_id];

    // not combining State and PRD as assume minimal PRD advantage exceeds copying cost 

    unsigned long long seqhis(0) ;
    unsigned long long seqmat(0) ;

    int MaterialIndex(0) ; 

    State s ;   
    Photon p ;  

    if(ghead.i.x == CERENKOV)   // 1st 4 bytes, is enumeration distinguishing cerenkov/scintillation/torch/...
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) csdebug(cs);
#endif
        generate_cerenkov_photon(p, cs, rng );         
        MaterialIndex = cs.MaterialIndex ;  
        s.flag = CERENKOV ;  
    }
    else if(ghead.i.x == SCINTILLATION)
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) ssdebug(ss);
#endif
        generate_scintillation_photon(p, ss, rng );         
        MaterialIndex = ss.MaterialIndex ;  
        s.flag = SCINTILLATION ;  
    }
    else if(ghead.i.x == TORCH)
    {
        TorchStep ts ;
        tsload(ts, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) tsdebug(ts);
#endif
        generate_torch_photon(p, ts, rng );         
        MaterialIndex = ts.MaterialIndex ;  
        s.flag = TORCH ;  
    }


    //p.flags.u.y = photon_id ;  // debug

    int bounce = 0 ; 
    int command = START ; 

    int slot = 0 ;
    int slot_min = photon_id*MAXREC ; 
    int slot_max = slot_min + MAXREC - 1 ; 
    int slot_offset = 0 ; 

    PerRayData_propagate prd ;

    while( bounce < bounce_max )
    {
        bounce++;   // increment at head, not tail, as CONTINUE skips the tail

        // trace sets these, see material1_propagate.cu:closest_hit_propagate
        prd.distance_to_boundary = -1.f ;
        prd.identity.x = 0 ; // nodeIndex
        prd.identity.y = 0 ; // meshIndex
        prd.identity.z = 0 ; // boundaryIndex, 0-based 
        prd.identity.w = 0 ; // sensorIndex
        prd.boundary = 0 ;   // signed, 1-based

        rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );


        if(prd.boundary == 0)
        {
            s.flag = MISS ;  // overwrite CERENKOV/SCINTILLATION for the no hitters
            // zero out no-hitters to avoid leftovers 
            s.index.x = 0 ;  
            s.index.y = 0 ;  
            s.index.z = 0 ; 
            s.index.w = 0 ; 
            break ;
        }   
        // initial and CONTINUE-ing records

        p.flags.i.x = prd.boundary ;  

        // use boundary index at intersection point to do optical constant + material/surface property lookups 
        fill_state(s, prd.boundary, prd.identity, p.wavelength );

        s.distance_to_boundary = prd.distance_to_boundary ; 
        s.surface_normal = prd.surface_normal ; 
        s.cos_theta = prd.cos_theta ; 

        p.flags.u.z = s.index.x ;   // material1 index 

        p.flags.u.w |= s.flag ; 


        if(dbg)
        {
           rtPrintf("bounce %d \n", bounce);
           rtPrintf("post  %10.3f %10.3f %10.3f %10.3f  % \n", p.position.x, p.position.y, p.position.z, p.time );
           rtPrintf("polw  %10.3f %10.3f %10.3f %10.3f  % \n", p.polarization.x, p.polarization.y, p.polarization.z, p.wavelength );
        } 


        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
        RSAVE(seqhis, seqmat, p, s, slot, slot_offset) ;
#ifdef AUX
        ASAVE(p, s, slot, slot_offset, MaterialIndex );
#endif
        slot++ ; 


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
        // PASS : survivors will go on to pick up one of the below flags, 
      

        if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
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


    //p.flags.u.y = photon_id ;      // photon_id matches the position in photon array
    //p.flags.u.y =  s.identity.x ;  // nodeIndex
    //p.flags.u.y =  s.identity.y ;  // meshIndex
    //p.flags.u.y =  s.identity.z ;  // boundaryIndex 0-based
    p.flags.u.y =  s.identity.w ;  // sensorIndex  >0 only for cathode hits



    psave(p, photon_buffer, photon_offset ); 

    // building history sequence, bounce by bounce
    // s.flag is unexpected coming out 0xf 20% of time ? always here on the last placement ?
    // skipping the 2nd seqset :
    //    0) avoids the problem of trailing f 
    //    1) causes all the MISS to get zero, the initial seqhis value
    //

    slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
    RSAVE(seqhis, seqmat, p, s, slot, slot_offset ) ;

#ifdef AUX
    ASAVE(p, s, slot, slot_offset, MaterialIndex);
#endif

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  

    rng_states[photon_id] = rng ;
}




RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}






