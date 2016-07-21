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

#include "define.h"
//#define DEBUG 1 
#include "PerRayData_propagate.h"

using namespace optix;

#include "quad.h"
#include "wavelength_lookup.h"

rtBuffer<uint4>                optical_buffer; 

#include "state.h"
#include "photon.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

#define GNUMQUAD 6

#include "cerenkovstep.h"
#include "scintillationstep.h"
#include "torchstep.h"

#include "rayleigh.h"
#include "propagate.h"

rtBuffer<float4>               genstep_buffer;
rtBuffer<float4>               photon_buffer;

// TODO: find no-compromise way to flip these switches without recompilation 
#define RECORD 1
#ifdef RECORD
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
#endif

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

// beyond MAXREC overwrite save into top slot
// TODO: check shift = slot_offset*4 rather than slot*4  ? 
 
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


RT_PROGRAM void trivial()
{
   //wavelength_dump(24, 10);    
   //wavelength_dump(42, 10);    
   //wavelength_dump(48, 10);    
   //wavelength_dump(1048, 10);    

    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    rtPrintf("(trivial) photon_id %d photon_offset %d \n", photon_id, photon_offset );

    // first 4 bytes of photon_buffer photon records is seeded with genstep_id 
    // this seeding is done by App::seedPhotonsFromGensteps
/*
    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;
    unsigned int genstep_id = phead.u.x ; 
    // getting crazy values for this in interop, photon_buffer being overwritten ?? 
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

    rtPrintf("(trivial) photon_id %d photon_offset %d genstep_id %d GNUMQUAD %d genstep_offset %d \n", photon_id, photon_offset, genstep_id, GNUMQUAD, genstep_offset  );

*/
    //rtPrintf("ghead.i.x %d \n", ghead.i.x );
    //
    // in interop mode ( GGeoViewTest --trivial ) on SDU Dell Precision Workstation getting genstep_id -1 
    // this causes an attempted read beyond the genstep buffer resuling on crash when accessing ghead.i.x
    //     (trivial) photon_id 4160 photon_offset 16640 genstep_id -1 GNUMQUAD 6 genstep_offset -6 
    //
    // in compute mode ( GGeoViewTest --trivial --compute ) get sensible genstep_id
    //     (trivial) photon_id 1057 photon_offset 4228 genstep_id 0 GNUMQUAD 6 genstep_offset 0 
    //
    //

/*
From compute mode run::

    2016-07-21 11:29:22.326 INFO  [9380] [OpEngine::preparePropagator@89] OpEngine::preparePropagator DONE 
    2016-07-21 11:29:22.326 INFO  [9380] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep name genstep size 6 multiplicity 4 sizeofatom 4 NumAtoms 24 NumBytes 96 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs : dev_ptr 0xb07200000 size 6 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon  name photon size 400000 multiplicity 4 sizeofatom 4 NumAtoms 1600000 NumBytes 6400000 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0xb07200000 size 6 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 
    2016-07-21 11:29:22.328 INFO  [9380] [OpSeeder::seedPhotonsFromGenstepsImp@141] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    TBufPair<T>::seedDestination (CBufSlice)src : dev_ptr 0xb07200000 size 6 num_bytes 96 stride 24 begin 3 end 24 
    TBufPair<T>::seedDestination (CBufSlice)dst : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 stride 16 begin 0 end 1600000 
    iexpand  counts_size 1 output_size 100000
    2016-07-21 11:29:22.332 INFO  [9380] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords


From interop mode run (OpSeeder buffer sizes are x4 ???)::

    2016-07-21 10:00:50.232 INFO  [881] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@79] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    CResource::mapGLToCUDA buffer_id 16 imp.bufsize 96      sizeof(T) 4 size 24 
    CResource::mapGLToCUDA buffer_id 18 imp.bufsize 6400000 sizeof(T) 4 size 1600000 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0x20491ae00 size 24 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0x20492f800 size 1600000 num_bytes 6400000 
    2016-07-21 10:00:50.239 INFO  [881] [OpSeeder::seedPhotonsFromGenstepsImp@134] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    TBufPair<T>::seedDestination (CBufSlice)src : dev_ptr 0x20491ae00 size 24 num_bytes 96 stride 24 begin 3 end 24 
    TBufPair<T>::seedDestination (CBufSlice)dst : dev_ptr 0x20492f800 size 1600000 num_bytes 6400000 stride 16 begin 0 end 1600000 
    iexpand  counts_size 1 output_size 100000
    2016-07-21 10:00:50.263 INFO  [881] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords

Source of the unexpected x4 bufsize is CResource::mapGLToCUDA

     53    void* mapGLToCUDA()
     54    {
     55        checkCudaErrors( cudaGraphicsMapResources(1, &resource, stream) );
     56        checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &bufsize, resource) );
     57        //printf("Resource::mapGLToCUDA bufsize %lu dev_ptr %p \n", bufsize, dev_ptr );
     58        return dev_ptr ;
     59    }

::

    In [3]: t = np.load("torchdbg.npy")

    In [9]: np.set_printoptions(suppress=True)

    In [10]: t
    Out[10]: 
    array([[[      0.      ,       0.      ,       0.      ,       0.      ],
            [ -18079.453125, -799699.4375  ,   -6605.      ,       0.1     ],
            [      0.      ,       0.      ,       1.      ,       1.      ],
            [      0.      ,       0.      ,       0.      ,     380.      ],
            [      0.      ,       1.      ,       0.      ,       1.      ],
            [      0.      ,       0.      ,       0.      ,       0.      ]]], dtype=float32)

    In [11]: t.view(np.int32)
    Out[11]: 
    array([[[      4096,          0,         95,     100000],
            [-963821848, -918340297, -976328704, 1036831949],
            [         0,          0, 1065353216, 1065353216],
            [         0,          0,          0, 1136525312],
            [         0, 1065353216,          0, 1065353216],
            [         0,          0,          0,          1]]], dtype=int32)

    In [5]: t.shape
    Out[5]: (1, 6, 4)

    In [6]: 6*4
    Out[6]: 24

    In [7]: 6*4*4
    Out[7]: 96      ## 96 bytes is correct




*/

}


RT_PROGRAM void generate()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 

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
        //MaterialIndex = cs.MaterialIndex ;  
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
        //MaterialIndex = ss.MaterialIndex ;  
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
        //MaterialIndex = ts.MaterialIndex ;  
        s.flag = TORCH ;  
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

    c4.uchar_.y = 2u ; 
    c4.uchar_.z = 3u ; 
    c4.uchar_.w = 4u ; 

    p.flags.f.z = c4.f ; 


    int bounce = 0 ; 
    int command = START ; 
    int slot = 0 ;

#ifdef RECORD
    unsigned long long seqhis(0) ;
    unsigned long long seqmat(0) ;
    int MaterialIndex(0) ; 
    unsigned int MAXREC = record_max ; 
    int slot_min = photon_id*MAXREC ; 
    int slot_max = slot_min + MAXREC - 1 ; 
    int slot_offset = 0 ; 
#endif

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

#ifdef RECORD
        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
        RSAVE(seqhis, seqmat, p, s, slot, slot_offset) ;
#endif

#ifdef AUX
        if(dbg)
        {
           rtPrintf("bounce %d \n", bounce);
           rtPrintf("post  %10.3f %10.3f %10.3f %10.3f  % \n", p.position.x, p.position.y, p.position.z, p.time );
           rtPrintf("polw  %10.3f %10.3f %10.3f %10.3f  % \n", p.polarization.x, p.polarization.y, p.polarization.z, p.wavelength );
        } 
        ASAVE(p, s, slot, slot_offset, MaterialIndex );
#endif
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

    }   // bounce < max_bounce


    FLAGS(p, s, prd); 

    // breakers and maxers saved here
    psave(p, photon_buffer, photon_offset ); 

#ifdef RECORD
    slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
    RSAVE(seqhis, seqmat, p, s, slot, slot_offset ) ;

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  
#endif


#ifdef AUX
    ASAVE(p, s, slot, slot_offset, MaterialIndex);
#endif

    rng_states[photon_id] = rng ;
}

RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}

