/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <curand_kernel.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "OpticksSwitches.h"

#ifdef ANGULAR_ENABLED
// Additional f_theta f_phi 
#include "PerRayData_angular_propagate.h"
#else
#include "PerRayData_propagate.h"
#endif

using namespace optix;

#ifdef WITH_SENSORLIB
// OSensorLib_* buffers and functions 
#include "OSensorLib.hh"
#endif

rtDeclareVariable(unsigned int,  PNUMQUAD, , );
rtDeclareVariable(unsigned int,  RNUMQUAD, , );
rtDeclareVariable(unsigned int,  GNUMQUAD, , );
rtDeclareVariable(unsigned int,  WNUMQUAD, , );


#include "quad.h"
#include "boundary_lookup.h"
#include "wavelength_lookup.h"

rtBuffer<uint4>                optical_buffer; 

// state.h struct embodied here as preprocessor.py switch control applies to .cu, not the .h included 
struct State 
{
   unsigned int flag ; 
   float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
   float4 m1group2  ;    // group_velocity/spare1/spare2/spare3
   float4 material2 ;  
   float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
   float3 surface_normal ; 
   float distance_to_boundary ;
   uint4 optical ;   // x/y/z/w index/type/finish/value  
   uint4 index ;     // indices of m1/m2/surf/sensor
   uint4 identity ;  //  node/mesh/boundary/sensor indices of last intersection

#ifdef WITH_REFLECT_CHEAT_DEBUG
   float ureflectcheat ;  
#endif

#ifdef WAY_ENABLED
   float4 way0 ;  
   float4 way1 ;  
#endif

};

#include "state.h"


#include "photon.h"
#include "OpticksGenstep.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

#include "cerenkovstep.h"

#include "scintillationstep.h"
#include "Genstep_G4Scintillation_1042.h"

#include "torchstep.h"

#include "rayleigh.h"
#include "propagate.h"


// input buffers 

rtBuffer<float4>               genstep_buffer;
rtBuffer<float4>               source_buffer;
rtBuffer<unsigned>             seed_buffer ; 
rtBuffer<curandState, 1>       rng_states ;

// output buffers 

rtBuffer<float4>               photon_buffer;
#ifdef WITH_RECORD
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
#endif

#ifdef WITH_DEBUG_BUFFER
rtBuffer<float4>               debug_buffer;
#endif
#ifdef WAY_ENABLED
rtBuffer<float4>               way_buffer;
#endif




rtDeclareVariable(float4,        center_extent, , );
rtDeclareVariable(float4,        time_domain  , , );
rtDeclareVariable(uint4,         debug_control , , );
rtDeclareVariable(float,         propagate_epsilon, , );

rtDeclareVariable(unsigned int,  propagate_ray_type, , );
rtDeclareVariable(unsigned int,  utaildebug, , );
rtDeclareVariable(unsigned int,  production, , );
rtDeclareVariable(unsigned int,  bounce_max, , );
rtDeclareVariable(unsigned int,  record_max, , );

rtDeclareVariable(int4,          way_control, , );


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


/**0
generate.cu
==============

* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/cu/generate.cu

.. contents:: Table of Contents
   :depth: 2

0**/


/**1
RSAVE Macro
------------

Writes compressed step points into the record buffer

Two in/outputs seqhis and seqmat, the rest are inputs
specifying sources of the data and where to write it. 

seqhis
    shifts in "his" nibble obtained from ffs of s.flag 
seqmat
    shifts in "mat" nibble obtained from s.index.x


1**/
 
#define RSAVE(seqhis, seqmat, p, s, slot, slot_offset)  \
{    \
    unsigned int shift = slot*4 ; \
    unsigned long long his = __ffs((s).flag) & 0xF ; \
    unsigned long long mat = (s).index.x < 0xF ? (s).index.x : 0xF ; \
    seqhis |= his << shift ; \
    seqmat |= mat << shift ; \
    rsave((p), (s).flag, (s).index, record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
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
        


/**2
FLAGS Macro 
------------

Sets the photon flags p.flags using values from state s and per-ray-data prd

p.flags.u.x 
   packed signed int boundary and unsigned sensorIndex which are 
   assumed to fit in 16 bits into 32 bits, see SPack::unsigned_as_int 

p.flags.u.y
   now getting s.identity.x (nodeIndex) thanks to the packing 

s.identity.x
    node index 

s.identity.w 
    sensor index arriving from GVolume::getIdentity.w

::

    256 glm::uvec4 GVolume::getIdentity() const
    257 {
    258     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    259     return id ;
    260 }

NumPy array access::

    boundary    = (( flags[:,0].view(np.uint32) & 0xffff0000 ) >> 16 ).view(np.int16)[1::2] 
    sensorIndex = (( flags[:,0].view(np.uint32) & 0x0000ffff ) >>  0 ).view(np.int16)[0::2] 


Formerly::

    p.flags.i.x = prd.boundary ;  \
    p.flags.u.y = s.identity.w ;  \
    p.flags.u.w |= s.flag ; \

2**/

#define FLAGS(p, s, prd) \
{ \
    p.flags.u.x = ( ((prd.boundary & 0xffff) << 16) | (s.identity.w & 0xffff) )  ;  \
    p.flags.u.y = s.identity.x ;  \
    p.flags.u.w |= s.flag ; \
} \



/*
nothing
---------

Used for debug swapping functions.

*/

RT_PROGRAM void nothing()
{
}

// BIZARRE NEW rtPrintf BUG IN OptiX 400 when formatting more than a single value
//
//   * first value prints with expected value
//   * second value appears as zero no matter what the real value
//   * third value appears as same at the first value no matter what the real value
// 

/*
dumpseed
---------

Just dumps the genstep_id for this photon_id obtained from the seed_buffer.

*/

RT_PROGRAM void dumpseed()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = unsigned(photon_id)*PNUMQUAD ; 

    unsigned int genstep_id = seed_buffer[photon_id] ;      
    rtPrintf("(dumpseed) genstep_id %u \n", genstep_id );

    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 
 
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


/*
trivial
--------

"trivial" goes one step beyond "dumpseed" in that 
it attempts to read from the genstep buffer

*/

RT_PROGRAM void trivial()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = unsigned(photon_id)*PNUMQUAD ; 
#ifdef WAY_ENABLED
    const unsigned int way_offset = photon_id*WNUMQUAD ; 
#endif
    unsigned int genstep_id = seed_buffer[photon_id] ;      
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 


    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 

    int gencode = ghead.i.x ; 

    rtPrintf("//(trivial) genstep_id %u genstep_offset %u gencode %d \n", genstep_id, genstep_offset, gencode );
   

    unsigned trivial_flags = photon_id % 1000 == 0 ? SURFACE_DETECT : 0u ; 

    quad indices ;  
    indices.u.x = photon_id ; 
    indices.u.y = photon_offset ; 
    indices.u.z = genstep_id ; 
    indices.u.w = trivial_flags ; 

    photon_buffer[photon_offset+0] = make_float4(  0.f , 0.f , 0.f, 0.f );
    photon_buffer[photon_offset+1] = make_float4(  0.f , 0.f , 0.f, 0.f );
    photon_buffer[photon_offset+2] = make_float4(  ghead.f.x,     ghead.f.y,    ghead.f.z,     ghead.f.w); 
    photon_buffer[photon_offset+3] = make_float4(  indices.f.x,   indices.f.y,  indices.f.z,   indices.f.w); 

#ifdef WAY_ENABLED
    way_buffer[way_offset+0] = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
    way_buffer[way_offset+1] = make_float4(  indices.f.x,   indices.f.y,  indices.f.z,   indices.f.w);  
#endif

#ifdef WITH_DEBUG_BUFFER
    debug_buffer[photon_offset+0] = make_float4(  0.f , 0.f , 0.f, 0.f );
#endif

 
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


/*
zrngtest
---------

Testing curand_uniform into the photon buffer.

*/

RT_PROGRAM void zrngtest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 

    curandState rng = rng_states[photon_id];

    photon_buffer[photon_offset+0] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+1] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+2] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    photon_buffer[photon_offset+3] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );

    //rng_states[photon_id] = rng ;  // suspect this does nothing in my usage  <-- and it plants 3 f64 in the PTX
}


/*
tracetest
----------

Test a single call to rtTrace

*/

RT_PROGRAM void tracetest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    unsigned int genstep_id = seed_buffer[photon_id] ;      
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    curandState rng = rng_states[photon_id];

    Photon p ;  

    TorchStep ts ;
    tsload(ts, genstep_buffer, genstep_offset, genstep_id);
    //tsdebug(ts);
    generate_torch_photon(p, ts, rng );         

#ifdef ANGULAR_ENABLED
    PerRayData_angular_propagate prd ;
#else 
    PerRayData_propagate prd ;
#endif

    // trace sets these, see material1_propagate.cu:closest_hit_propagate
    prd.distance_to_boundary = -1.f ;

    prd.identity.x = 0 ; 
    prd.identity.y = 0 ;
    prd.identity.z = 0 ;
    prd.identity.w = 0 ;

    prd.boundary = 0 ;   // signed, 1-based

    rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );

    p.flags.u.x = prd.identity.x ;   // nodeIndex               -> nodeIndex   
    p.flags.u.y = prd.identity.y ;   // meshIndex               -> tripletIdentity
    p.flags.u.z = prd.identity.z ;   // boundaryIndex, 0-based  -> SPack::Encode22(meshIndex,boundaryIndex)
    p.flags.u.w = prd.identity.w ;   // sensorIndex             -> sensorIndex

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



/**3
generate.cu:generate : generate + propagate filling photon_buffer 
-------------------------------------------------------------------

Prior to bounce loop 
~~~~~~~~~~~~~~~~~~~~~~~

* get genstep_id from seed_buffer and access the genstep for the photon_id 
* generate Cerenkov/Scintillation/Torch photons using implementations 
  from the corresponding headers
  
Inside bounce loop before record save 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

                            /\
     *--> . . . . . . m1  ./  \ - m2 - - - -
    p                     / su \
                         /______\


* rtTrace finds intersect along ray from p.position in p.direction, 
  and geometry closest hit function sets prd.boundary  

* prd.boundary and p.wavelength used to lookup surface/material properties 
  from the boundary texture

* note that so far there is no change to **p.direction/p.position/p.time** 
  they are still at their generated OR last changed positions 
  from CONTINUE-ers from the below propagation code 

In summary before record save are looking ahead to see where 
to go next while the photon remains with the existing position, 
direction and flags from the prior "step" or generation.


Inside bounce loop after record save
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* changes p.position p.time p.direction p.polarization ... 
  then CONTINUEs back to the above to find the next intersection
  or BREAKs


::

    .       BULK_SCATTER/BULK_REEMIT
                ^
                |           /\
     *--> . # . *. . . m1  /  \ - m2 - - - -
    p       .             / su \
         BULK_ABSORB     /______\

* each turn of the loop only sets a single history bit  
* bit position is used rather than the full mask in the photon record
  to save bits  

* control the or-ing of that flag into photon history at this level
* integrate surface optical props: finish=polished/ground

3**/


RT_PROGRAM void generate()
{
    unsigned long long photon_id = launch_index.x ;  
#ifdef WITH_REFLECT_CHEAT_DEBUG
    unsigned long long num_photon = launch_dim.x ;
#endif
  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
#ifdef WAY_ENABLED
    const unsigned int way_offset = photon_id*WNUMQUAD ; 
#endif
    unsigned int genstep_id = seed_buffer[photon_id] ;      
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    union quad ghead ; 
    ghead.f = genstep_buffer[genstep_offset+0]; 
    int gencode = ghead.i.x ;  // 1st 4 bytes, is enumeration distinguishing cerenkov/scintillation/torch/...

#ifdef DEBUG
    bool dbg = photon_id == debug_control.x ;  
    if(dbg)
    {
       rtPrintf("generate debug photon_id %d genstep_id %d ghead.i.x %d \n", photon_id, genstep_id, ghead.i.x );
    } 
    rtPrintf("generate photon_id %d \n", photon_id );
#endif 

    curandState rng = rng_states[photon_id];

    State s ;   
    Photon p ;  

#ifdef WITH_REFLECT_CHEAT_DEBUG
    s.ureflectcheat = 0.f ; 
#endif

    if(gencode == OpticksGenstep_G4Cerenkov_1042 ) 
    {
        CerenkovStep cs ;
        csload(cs, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) csdebug(cs);
#endif
        generate_cerenkov_photon(p, cs, rng );         
        s.flag = CERENKOV ;  
    }
    else if(gencode == OpticksGenstep_DsG4Scintillation_r3971 )
    {
        ScintillationStep ss ;
        ssload(ss, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) ssdebug(ss);
#endif
        generate_scintillation_photon(p, ss, rng );  // maybe split on gencode ?
        s.flag = SCINTILLATION ;  
    }
    else if(gencode == OpticksGenstep_G4Scintillation_1042 )
    {
        Genstep_G4Scintillation_1042 ss ;
        ss.load( genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) ss.debug();
#endif
        ss.generate_photon(p, rng ); 
        s.flag = SCINTILLATION ;  
    }
    else if(gencode == OpticksGenstep_TORCH)
    {
        TorchStep ts ;
        tsload(ts, genstep_buffer, genstep_offset, genstep_id);
#ifdef DEBUG
        if(dbg) tsdebug(ts);
#endif
        generate_torch_photon(p, ts, rng );         
        s.flag = TORCH ;  
    }
    else if(gencode == OpticksGenstep_EMITSOURCE)
    {
        // source_buffer is input only, photon_buffer output only, 
        // photon_offset is same for both these buffers
        pload(p, source_buffer, photon_offset ); 
        s.flag = TORCH ;  
#ifdef WITH_REFLECT_CHEAT_DEBUG
        s.ureflectcheat = debug_control.w > 0u ? float(photon_id)/float(num_photon) : -1.f ;
#endif
    }

#ifdef WAY_ENABLED
/**
Contents of way buffer must correspond to the ggeo/GPho/getWay methods. 
The final values are fed into *way* buffer which gets selected down to *hiy* array 
which ends up in G4OpticksHitExtra. Note the *way* to *hiy* is precisely 
analogous to *photon* to *hit* selexction.
**/
    s.way0.x = 0.f ;                          // boundary_pos.x   (mm)
    s.way0.y = 0.f ;                          // boundary_pos.y   (mm)
    s.way0.z = 0.f ;                          // boundary_pos.z   (mm)
    s.way0.w = 0.f ;                          // boundary_time    (ns)

    s.way1.x = p.time ;                       // origin_time of generated photon (ns)
    s.way1.y = int_as_float( ghead.i.y ) ;    // origin_trackID  from genstep  0*4+1 which is G4Track::GetTrackID of parent
    s.way1.z = 0.f ;                          // p.flags.u.z (photon_index) is duplicated here for debugging post selection
    s.way1.w = 0.f ;                          // p.flags.u.w (bitwise-or-of-step-flags) is duplicated here for hiy selection
#endif

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

    if( production == 0 )
    {
        for(slot=0 ; slot < MAXREC ; slot++)
        {
             record_offset = (slot_min + slot)*RNUMQUAD ;
             record_buffer[record_offset+0] = make_short4(0,0,0,0) ;    // 4*int16 = 64 bits
             record_buffer[record_offset+1] = make_short4(0,0,0,0) ;    
        }
    }
    slot = 0 ; 
    record_offset = 0 ; 
#endif



#ifdef ANGULAR_ENABLED
    PerRayData_angular_propagate prd ;
#else
    PerRayData_propagate prd ;
#endif

#ifdef WITH_DEBUG_BUFFER
    prd.debug.x = 0.f ; 
    prd.debug.y = 0.f ; 
    prd.debug.z = 0.f ; 
#endif

    while( bounce < bounce_max )
    {
#ifdef WITH_ALIGN_DEV_DEBUG
        rtPrintf("WITH_ALIGN_DEV_DEBUG photon_id:%d bounce:%d \n", photon_id, bounce );
#endif

        bounce++;   // increment at head, not tail, as CONTINUE skips the tail

        // trace sets these, see material1_propagate.cu:closest_hit_propagate
        prd.distance_to_boundary = -1.f ;
        prd.identity.x = 0 ; // nodeIndex                -> nodeIndex
        prd.identity.y = 0 ; // meshIndex                -> tripletIdentity
        prd.identity.z = 0 ; // boundaryIndex, 0-based   -> SPack::Encode22(meshIndex, boundaryIndex)
        prd.identity.w = 0 ; // sensorIndex              -> sensorIndex
        prd.boundary = 0 ;   // signed, 1-based


        // TODO: minimize active stack across the rtTrace call
        rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );

//#define WITH_PRINT_IDENTITY 1 
#ifdef WITH_PRINT_IDENTITY
        rtPrintf("//generate.cu WITH_PRINT_IDENTITY prd.identity ( %8d %8d %8d %8d )\n", prd.identity.x, prd.identity.y, prd.identity.z, prd.identity.w); 
#endif

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

        // setting p.flags for things like boundary, history flags  
        FLAGS(p, s, prd); 

#ifdef WITH_RECORD
        if( production == 0 )
        {
            slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
            RSAVE(seqhis, seqmat, p, s, slot, slot_offset) ;
        } 
#endif

        slot++ ; 

        command = propagate_to_boundary( p, s, rng );
        if(command == BREAK)    break ;           // BULK_ABSORB
        if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER

        // tacit PASS : survivors succeed to reach the boundary 
        // proceeding to pick up one of the below SURFACE_ or BOUNDARY_ flags 

#ifdef WAY_ENABLED
        //if( way_control.x == prd.identity.x && way_control.y == prd.boundary )
        if( way_control.y == prd.boundary )
        {
            s.way0.x = p.position.x ; 
            s.way0.y = p.position.y ; 
            s.way0.z = p.position.z ; 
            s.way0.w = p.time ; 
        }
#endif
        if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
        {
            command = propagate_at_surface(p, s, rng);
            if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
            if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
        }
        else
        {
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



#ifdef WITH_SENSORLIB
    if( s.flag == SURFACE_DETECT ) 
    {
        const unsigned& sensorIndex = s.identity.w ;   // should always be > 0 as flag is SD
#ifdef ANGULAR_ENABLED
        const float& f_theta = prd.f_theta ;
        const float& f_phi = prd.f_phi ; 
        const float efficiency = OSensorLib_combined_efficiency(sensorIndex, f_phi, f_theta);
        //rtPrintf("//SD sensorIndex %d f_theta %f f_phi %f efficiency %f \n", sensorIndex, f_theta, f_phi, efficiency );
#else
        const float efficiency = OSensorLib_simple_efficiency(sensorIndex);
        //rtPrintf("//SD sensorIndex %d efficiency %f \n", sensorIndex, efficiency );
#endif
        float u_angular = curand_uniform(&rng) ;
        p.flags.u.w |= ( u_angular < efficiency ?  EFFICIENCY_COLLECT : EFFICIENCY_CULL ) ;   
    } 
#endif


    // setting p.flags for things like boundary, history flags  
    FLAGS(p, s, prd); 

    p.flags.u.z = photon_id ;  // formerly behind IDENTITY_DEBUG macro, but has become indispensable

    if( utaildebug )   // --utaildebug    see notes/issues/ts-box-utaildebug-decouple-maligned-from-deviant.rst
    {
        float u_taildebug = curand_uniform(&rng) ;
        p.flags.f.y = u_taildebug ; 
#ifdef WITH_ALIGN_DEV_DEBUG
        rtPrintf("generate u_OpBoundary_taildebug:%.9g \n", u_taildebug ); 
#endif
    }


#ifdef WAY_ENABLED
    s.way1.z = unsigned_as_float(p.flags.u.z) ;  // photon index  
    s.way1.w = unsigned_as_float(p.flags.u.w) ;  // flags duplicated for hiy selection from way buffer (analogous to hit selection from photons buffer)
    way_buffer[way_offset+0] = s.way0 ; 
    way_buffer[way_offset+1] = s.way1 ; 
#endif

    //  s.identity.x : nodeIndex 
    //  s.identity.y : tripletIdentity
    //  s.identity.z : shapeIdentity
    //  s.identity.w : sensorIndex   (already in flags)
    //
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
    if( production == 0 )
    {
        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;  
        RSAVE(seqhis, seqmat, p, s, slot, slot_offset ) ;

        sequence_buffer[photon_id*2 + 0] = seqhis ; 
        sequence_buffer[photon_id*2 + 1] = seqmat ;  
    }
#endif


//#define WITH_PRINT_IDENTITY_GE 1
#ifdef WITH_PRINT_IDENTITY_GE
   //if(p.flags.i.w & SURFACE_DETECT )
   rtPrintf("// generate.cu WITH_PRINT_IDENTITY_GE  photon_id %d \n", photon_id ); 
#endif



    /**
     rng_states
          because the entire simulation of a single photon is done via this generate.cu 
          with a single curandState for each photon, there is no need to save the curandState 
          as there is no continuation 

          TODO: check what happens from event to event, when the same curandState gets used
          for another photon  
    **/
    //rng_states[photon_id] = rng ;  // <-- brings 3 lines with .f64 : is it needed ???

}




RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
#ifdef WITH_EXCEPTION
    rtPrintExceptionDetails();
#endif
    photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}

