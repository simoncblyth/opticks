// Where best to record the propagation ? 
// =======================================
//
// Code prior to inloop record save 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// * **NB THIS CODE DOES NOT change p.direction/p.position/p.time** 
//   they are still at their generated OR last changed positions 
//   from CONTINUE-ers from the below propagation code 
//
//  In summary this code is looking ahead to see where to go next
//  while the photon remains with exitant position and direction
//  and flags from the last "step"
//
// Code after inloop record save
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// * each turn of the loop only sets a single history bit  
// * bit position is used rather than the full mask in the photon record
//   to save bits  
//
// * control the or-ing of that flag into photon history at this level
// * integrate surface optical props: finish=polished/ground
//
//


#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

//#define DEBUG 1 
#include "PerRayData_propagate.h"
#include "OpticksSwitches.h"

using namespace optix;

//rtDeclareVariable(float,         SPEED_OF_LIGHT, , );
rtDeclareVariable(unsigned int,  PNUMQUAD, , );
rtDeclareVariable(unsigned int,  RNUMQUAD, , );
rtDeclareVariable(unsigned int,  GNUMQUAD, , );

#include "quad.h"
#include "boundary_lookup.h"
#include "wavelength_lookup.h"

rtBuffer<uint4>                optical_buffer; 

#include "state.h"
#include "photon.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//rtDeclareVariable(float, t_parameter, rtIntersectionDistance, );
// not giving prd.distance_to_boundary

#include "cerenkovstep.h"
#include "scintillationstep.h"
#include "torchstep.h"

#include "rayleigh.h"
#include "propagate.h"


// input buffers 

rtBuffer<float4>               genstep_buffer;
rtBuffer<float4>               source_buffer;
#ifdef WITH_SEED_BUFFER
rtBuffer<unsigned>             seed_buffer ; 
#endif
rtBuffer<curandState, 1>       rng_states ;

// output buffers 

rtBuffer<float4>               photon_buffer;
#ifdef WITH_RECORD
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
#endif




rtDeclareVariable(float4,        center_extent, , );
rtDeclareVariable(float4,        time_domain  , , );
rtDeclareVariable(uint4,         debug_control , , );
rtDeclareVariable(float,         propagate_epsilon, , );
rtDeclareVariable(unsigned int,  propagate_ray_type, , );
rtDeclareVariable(unsigned int,  bounce_max, , );
rtDeclareVariable(unsigned int,  record_max, , );

rtDeclareVariable(rtObject,      top_object, , );

// beyond MAXREC overwrite save into top slot
// TODO: check shift = slot_offset*4 rather than slot*4  ? 
//       nope slot_offset*RNUMQUAD is absolute pointing into record_buffer
//       so maybe
//             shift = ( slot < MAXREC ? slot : MAXREC - 1 )* 4 
//
//       slot_offset constraint  
//            int slot_min = photon_id*MAXREC ; 
//            int slot_max = slot_min + MAXREC - 1 ;
//            slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;
//
//        so in terms of saving location into record buffer, tis constrained correctly
//        BUT the seqhis shifts look wrong in truncation 
//
 
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
        

#define FLAGS(p, s, prd) \
{ \
    p.flags.i.x = prd.boundary ;  \
    p.flags.u.y = s.identity.w ;  \
    p.flags.u.w |= s.flag ; \
} \


//    p.flags.u.z = s.index.x ;   \



RT_PROGRAM void nothing()
{
}

// BIZARRE NEW rtPrintf BUG IN OptiX 400 when formatting more than a single value
//
//   * first value prints with expected value
//   * second value appears as zero no matter what the real value
//   * third value appears as same at the first value no matter what the real value
// 
// rtPrintf("(trivial) photon_id %u \n", photon_id );
// rtPrintf("(trivial) photon_offset %u \n", photon_offset );
// rtPrintf("(trivial) photon_id %u photon_id %u photon_offset %u \n", photon_id, photon_id, photon_offset );
//

RT_PROGRAM void dumpseed()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = unsigned(photon_id)*PNUMQUAD ; 

#ifdef WITH_SEED_BUFFER
    unsigned int genstep_id = seed_buffer[photon_id] ;      
    rtPrintf("(dumpseed WITH_SEED_BUFFER) genstep_id %u \n", genstep_id );
#else
    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;   
    unsigned int genstep_id = phead.u.x ;        
    rtPrintf("(dumpseed NOT with_seed_buffer) genstep_id %u \n", genstep_id );
#endif

    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 
    //rtPrintf("(trivial) genstep_offset %u \n", genstep_offset );
 
    quad indices ;  
    indices.u.x = photon_id ; 
    indices.u.y = photon_offset ; 
    indices.u.z = genstep_id ; 
    indices.u.w = genstep_offset ; 

    //photon_buffer[photon_offset+0] = make_float4(  0.f , 0.f , 0.f, 0.f );
    // writing over where the seeds were causes the problem for 2nd event
    // until moved to BUFFER_COPY_ON_DIRTY and manual markDirty
    //
    photon_buffer[photon_offset+3] = make_float4(  indices.f.x,   indices.f.y,  indices.f.z,   indices.f.w); 
}


RT_PROGRAM void trivial()
{
    // "trivial" goes one step beyond "dumpseed" in that 
    // it attempts to read from the genstep buffer

    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = unsigned(photon_id)*PNUMQUAD ; 

#ifdef WITH_SEED_BUFFER
    unsigned int genstep_id = seed_buffer[photon_id] ;      
#else
    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;   // crazy values for this in interop mode on Linux, photon_buffer being overwritten ?? 
    unsigned int genstep_id = phead.u.x ;        // input "seed" pointing from photon to genstep (see seedPhotonsFromGensteps)
#endif
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 


    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

    int gencode = ghead.i.x ; 

    rtPrintf("(trivial) genstep_id %u genstep_offset %u gencode %d \n", genstep_id, genstep_offset, gencode );


   
    quad indices ;  
    indices.u.x = photon_id ; 
    indices.u.y = photon_offset ; 
    indices.u.z = genstep_id ; 
    indices.u.w = genstep_offset ; 

    photon_buffer[photon_offset+0] = make_float4(  0.f , 0.f , 0.f, 0.f );
    photon_buffer[photon_offset+1] = make_float4(  0.f , 0.f , 0.f, 0.f );
    photon_buffer[photon_offset+2] = make_float4(  ghead.f.x,     ghead.f.y,    ghead.f.z,     ghead.f.w); 
    photon_buffer[photon_offset+3] = make_float4(  indices.f.x,   indices.f.y,  indices.f.z,   indices.f.w); 

 
    //rtPrintf("(trivial) GNUMQUAD %d PNUMQUAD %d RNUMQUAD %d \n", GNUMQUAD, PNUMQUAD, RNUMQUAD );
    //rtPrintf("(trivial) photon_id %u photon_offset %u genstep_id %u genstep_offset %u gencode %d \n", photon_id, photon_offset, genstep_id, genstep_offset, gencode );
    //rtPrintf("ghead.i.x %d \n", ghead.i.x );
    //
    // in interop mode ( GGeoViewTest --trivial ) on SDU Dell Precision Workstation getting genstep_id -1 
    // this causes an attempted read beyond the genstep buffer resuling on crash when accessing ghead.i.x
    //     (trivial) photon_id 4160 photon_offset 16640 genstep_id -1 GNUMQUAD 6 genstep_offset -6 
    //
    // in compute mode ( GGeoViewTest --trivial --compute ) get sensible genstep_id
    //     (trivial) photon_id 1057 photon_offset 4228 genstep_id 0 GNUMQUAD 6 genstep_offset 0 
    //
}



RT_PROGRAM void zrngtest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 

    curandState rng = rng_states[photon_id];

    photon_buffer[photon_offset+0] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+1] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+2] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+3] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );

    rng_states[photon_id] = rng ;  // suspect this does nothing in my usage
}




RT_PROGRAM void tracetest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    unsigned int genstep_id = seed_buffer[photon_id] ;      
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    //union quad ghead ; 
    //ghead.f = genstep_buffer[genstep_offset+0]; 
    //int gencode = ghead.i.x ; 

    curandState rng = rng_states[photon_id];

    //State s ;   
    Photon p ;  

    TorchStep ts ;
    tsload(ts, genstep_buffer, genstep_offset, genstep_id);
    //tsdebug(ts);
    generate_torch_photon(p, ts, rng );         

    //s.flag = TORCH ;  
 
    PerRayData_propagate prd ;

    // trace sets these, see material1_propagate.cu:closest_hit_propagate
    prd.distance_to_boundary = -1.f ;

    prd.identity.x = 0 ; // nodeIndex
    prd.identity.y = 0 ; // meshIndex
    prd.identity.z = 0 ; // boundaryIndex, 0-based 
    prd.identity.w = 0 ; // sensorIndex

    prd.boundary = 0 ;   // signed, 1-based

    rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );

    p.flags.u.x = prd.identity.x ; 
    p.flags.u.y = prd.identity.y ; 
    p.flags.u.z = prd.identity.z ; 
    p.flags.u.w = prd.identity.w ; 

/*
    rtPrintf("[%6d]tracetest distance_to_boundary %7.2f  id %4d %4d %4d %4d  boundary %4d  tpos %7.2f %7.2f %7.2f   cos_theta %7.2f \n", 
         launch_index.x, 
         prd.distance_to_boundary, 
         prd.identity.x, 
         prd.identity.y, 
         prd.identity.z, 
         prd.identity.w,
         prd.boundary,
         p.direction.x,
         p.direction.y,
         p.direction.z,
         prd.cos_theta
       );
*/

    union quad q ;
    q.i.w = prd.boundary ; 


    photon_buffer[photon_offset+0] = make_float4( p.position.x,     p.position.y,    p.position.z,     214.f  ); 
    photon_buffer[photon_offset+1] = make_float4( p.direction.x,    p.direction.y,   p.direction.z,     prd.distance_to_boundary ); 
    photon_buffer[photon_offset+2] = make_float4( prd.surface_normal.x, prd.surface_normal.y, prd.surface_normal.z , q.f.w  );
    photon_buffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w); 

}




RT_PROGRAM void generate()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned long long num_photon = launch_dim.x ;
  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 

#ifdef WITH_SEED_BUFFER
    // this is default 
    unsigned int genstep_id = seed_buffer[photon_id] ;      
#else
    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;   // crazy values for this in interop mode on Linux, photon_buffer being overwritten ?? 
    unsigned int genstep_id = phead.u.x ;        // input "seed" pointing from photon to genstep (see seedPhotonsFromGensteps)
#endif
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 
    int gencode = ghead.i.x ; 


#ifdef DEBUG
    bool dbg = photon_id == debug_control.x ;  
    if(dbg)
    {
       rtPrintf("generate debug photon_id %d genstep_id %d ghead.i.x %d \n", photon_id, genstep_id, ghead.i.x );
    } 
#endif 

    rtPrintf("generate photon_id %d \n", photon_id );

    curandState rng = rng_states[photon_id];

    State s ;   
    Photon p ;  

    s.ureflectcheat = 0.f ; 

    if(gencode == CERENKOV)   // 1st 4 bytes, is enumeration distinguishing cerenkov/scintillation/torch/...
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) csdebug(cs);
#endif
        generate_cerenkov_photon(p, cs, rng );         
        s.flag = CERENKOV ;  
    }
    else if(gencode == SCINTILLATION)
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) ssdebug(ss);
#endif
        generate_scintillation_photon(p, ss, rng );         
        s.flag = SCINTILLATION ;  
    }
    else if(gencode == TORCH)
    {
        TorchStep ts ;
        tsload(ts, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) tsdebug(ts);
#endif
        generate_torch_photon(p, ts, rng );         
        s.flag = TORCH ;  
    }
    else if(gencode == EMITSOURCE)
    {
        // source_buffer is input only, photon_buffer output only, 
        // photon_offset is same for both these buffers
        pload(p, source_buffer, photon_offset ); 
        s.flag = TORCH ;  
        s.ureflectcheat = debug_control.w > 0u ? float(photon_id)/float(num_photon) : -1.f ;
    }



    // initial quadrant 
    uifchar4 c4 ; 
    c4.uchar_.x = 
                  (  p.position.x > 0.f ? QX : 0u ) 
                   |
                  (  p.position.y > 0.f ? QY : 0u ) 
                   |
                  (  p.position.z > 0.f ? QZ : 0u )
                  ; 

    c4.uchar_.y = 2u ;   // 3-bytes up for grabs
    c4.uchar_.z = 3u ; 
    c4.uchar_.w = 4u ; 

    p.flags.f.z = c4.f ; 




    

    //rtPrintf("(generate.cu) p0 %10.4f %10.4f %10.4f  \n", p.position.x, p.position.y, p.position.z );

    int bounce = 0 ; 
    int command = START ; 
    int slot = 0 ;

#ifdef WITH_RECORD
    unsigned long long seqhis(0) ;
    unsigned long long seqmat(0) ;
    unsigned int MAXREC = record_max ; 
    int slot_min = photon_id*MAXREC ; 
    int slot_max = slot_min + MAXREC - 1 ; 
    int slot_offset = 0 ; 


    // zeroing record buffer, needed as OpZeroer not working in interop mode with OptiX 400
    int record_offset = 0 ; 
    for(slot=0 ; slot < MAXREC ; slot++)
    {
         record_offset = (slot_min + slot)*RNUMQUAD ;
         record_buffer[record_offset+0] = make_short4(0,0,0,0) ;    // 4*int16 = 64 bits
         record_buffer[record_offset+1] = make_short4(0,0,0,0) ;    
    }  
    slot = 0 ; 
    record_offset = 0 ; 
#endif




    PerRayData_propagate prd ;

    while( bounce < bounce_max )
    {
#ifdef WITH_ALIGN_DEV_DEBUG
        rtPrintf("WITH_ALIGN_DEV_DEBUG photon_id:%d bounce:%d \n", photon_id, bounce );
#endif

        bounce++;   // increment at head, not tail, as CONTINUE skips the tail

        // trace sets these, see material1_propagate.cu:closest_hit_propagate
        prd.distance_to_boundary = -1.f ;
        prd.identity.x = 0 ; // nodeIndex
        prd.identity.y = 0 ; // meshIndex
        prd.identity.z = 0 ; // boundaryIndex, 0-based 
        prd.identity.w = 0 ; // sensorIndex
        prd.boundary = 0 ;   // signed, 1-based


        // TODO: minimize active stack across the rtTrace call
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

        // use boundary index at intersection point to do optical constant + material/surface property lookups 
        fill_state(s, prd.boundary, prd.identity, p.wavelength );

        s.distance_to_boundary = prd.distance_to_boundary ; 
        s.surface_normal = prd.surface_normal ; 
        s.cos_theta = prd.cos_theta ; 

        FLAGS(p, s, prd); 

#ifdef WITH_RECORD
        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
        RSAVE(seqhis, seqmat, p, s, slot, slot_offset) ;
#endif


        //
        // see seqvol.rst 
        //     below dumping more useful with option --pindex 0/1/2 etc.. restricting indices to dump
        //     in order to follow a single photons intersect volumes
        //   
        // rtPrintf(" photon_id %d slot %d s.identity.x %d \n", photon_id, slot, s.identity.x );
        //

        slot++ ; 

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
            //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
            propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
            // tacit CONTINUE
        }

    }   // bounce < bounce_max


    // about to write non-BREAKER into topslot : this means truncation  
    //if( bounce == bounce_max && ( s.flag == SURFACE_DETECT || s.flag == SURFACE_ABSORB ))
    //if( bounce == bounce_max && command != BREAK )
    //if( bounce == bounce_max && !( s.flag == SURFACE_DREFLECT || s.flag == BULK_SCATTER || s.flag == BULK_REEMIT || s.flag == SURFACE_SREFLECT  ))

    if( bounce == bounce_max && s.flag == BOUNDARY_TRANSMIT )
    {
        s.index.x = s.index.y ;   // kludge putting m2->m1 for seqmat for the truncated
    }


    FLAGS(p, s, prd); 

    // breakers and maxers saved here
    psave(p, photon_buffer, photon_offset ); 

#ifdef WITH_ALIGN_DEV_DEBUG
    rtPrintf(" WITH_ALIGN_DEV_DEBUG psave (%.9g %.9g %.9g %.9g) ( %d, %d, %d, %d ) \n",
                 p.position.x,    p.position.y,    p.position.z,     p.time, 
                 p.flags.i.x ,    p.flags.i.y,     p.flags.i.z,    p.flags.i.w
         ); 
#endif

    // RSAVE lays down s.flag and s.index.x into the seqhis and seqmat
    // but there is inconsistency for BREAKers as s.index.x (m1) is only updated by fill_state 
    // but s.flag is updated after that by the propagate methods : so the last m1 
    // will usually be repeated in seqmat and the material on which the absorb or detect 
    // happened will be missed
    //
    //  kludged this with s.index.y -> s.index.x in propagate for SURFACE_ABSORB and SURFACE_DETECT
    //  BUT WHAT ABOUT TRUNCATION ? DONT GET TO THE BREAK ?
    //

#ifdef WITH_RECORD
    slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
    RSAVE(seqhis, seqmat, p, s, slot, slot_offset ) ;

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  
#endif


    rng_states[photon_id] = rng ;
}




RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}

