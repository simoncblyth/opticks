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

#pragma once

// see npy-/numutil.cpp
// http://stackoverflow.com/questions/7337526/how-to-tell-if-a-32-bit-int-can-fit-in-a-16-bit-short
// http://en.wikipedia.org/wiki/Two's_complement
// http://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers
// http://stereopsis.com/radix.html
// https://graphics.stanford.edu/~seander/bithacks.html
// http://www.informit.com/articles/article.aspx?p=2033340&seqNum=3

#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)

#include "OpticksPhoton.h"
#include "OpticksQuadrant.h"

enum { BREAK, CONTINUE, PASS, START, RETURN }; // return value from propagate_to_boundary


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define PHOTON_METHOD __device__
#else
   #define PHOTON_METHOD 
#endif 


struct Photon
{
   float3 position ;  
   float  time ;  

   float3 direction ;
   float  weight ; 

   float3 polarization ;
   float  wavelength ; 

   quad flags ;  

   // flags.i.x : boundary index (1-based signed, 0: no intersect)     [easily 16 bits]
   // flags.u.y : sensor index (signed, -1 means non-sensor)           [easily 16 bits]
   // flags.u.z : photon index (origin index useful after selections)  [needs 32 bits]
   // flags.u.w : history mask (bitwise OR of all step flags)          [could manage 16 bits but complicated as used to define hits]
                   
};


// optix::buffer<float4>& pbuffer
PHOTON_METHOD void pload( Photon& p, const float4* pbuffer, unsigned int photon_offset)
{
    const float4& post = pbuffer[photon_offset+0];
    const float4& dirw = pbuffer[photon_offset+1];
    const float4& polw = pbuffer[photon_offset+2];
    const float4& flags = pbuffer[photon_offset+3];

    p.position.x = post.x ; 
    p.position.y = post.y ; 
    p.position.z = post.z ;
    p.time = post.w ;

    p.direction.x = dirw.x ; 
    p.direction.y = dirw.y ; 
    p.direction.z = dirw.z ;
    p.weight = dirw.w ;

    p.polarization.x = polw.x ; 
    p.polarization.y = polw.y ; 
    p.polarization.z = polw.z ;
    p.wavelength = polw.w ;

    p.flags.f.x = flags.x ; 
    p.flags.f.y = flags.y ; 
    p.flags.f.z = flags.z ; 
    p.flags.f.w = flags.w ; 
}


// optix::buffer<float4>& pbuffer
PHOTON_METHOD void psave( Photon& p, float4* pbuffer, unsigned int photon_offset)
{
    pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time ); 
    pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
    pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
    pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w); 
}




/**
shortnorm
------------

range of short is -32768 to 32767
Expect no positions out of range, as constrained by the geometry are bouncing on,
but getting times beyond the range eg 0.:100 ns is expected

**/

PHOTON_METHOD short shortnorm( float v, float center, float extent )
{
    int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 

// optix::buffer<short4>& rbuffer
PHOTON_METHOD void rsave_zero( short4* rbuffer, unsigned int record_offset )
{
    rbuffer[record_offset+0] = make_short4(0,0,0,0) ;    // 4*int16 = 64 bits
    rbuffer[record_offset+1] = make_short4(0,0,0,0) ;    
}



/**
rsave
--------

1. packs position and time into normalized shorts (4*16 = 64 bits)

   * NB shortnorm wastes half the bits for time, as do not get negative times,  
     TODO: use a "ushortnorm" : THIS IS NOT WORTH IT AS LOTS OF ANALYSIS RELIES
     ON TIME AND POS USING THE SAME COMPRESSION APPROACH  

2. pack polarization and wavelength into 4*8 = 32 bits   

   * polarization already normalized into -1.f:1.f
   * wavelenth normalized via  (wavelength - low)/range into 0.:1. 
   * range of char -128:127  normalization of polarization and wavelength expected bulletproof, so no handling of out-of-range 
   * range of uchar 0:255   -1.f:1.f  + 1 => 0.f:2.f  so scale by 127.f 
 
POSSIBLE ALTERNATIVE: use a more vectorized approach, ie
   
* combine position and time domains into single float4 on the host 
* after verification can dispense with the fit checking for positions, just do time
* adopt p.position_time  maybe p.polarization_wavelength
* simularly with domains of those ?

**/


// optix::buffer<short4>& rbuffer

PHOTON_METHOD void rsave( Photon& p, unsigned s_flag, uint4& s_index, short4* rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain, float4& boundary_domain )
{
    rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
                    shortnorm(p.position.x, center_extent.x, center_extent.w), 
                    shortnorm(p.position.y, center_extent.y, center_extent.w), 
                    shortnorm(p.position.z, center_extent.z, center_extent.w),   
                    shortnorm(p.time      , time_domain.x  , time_domain.y  )
                    ); 

    float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 

    qquad qpolw ;    
    qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;  // pol : -1->1  pol+1 : 0->2   (pol+1)*127 : 0->254
    qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;

    // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 
    hquad polw ;    // union of short4, ushort4
    polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;
    

#ifdef IDENTITY_CHECK
    // spread uint32 photon_id across two uint16
    unsigned int photon_id = p.flags.u.y ;
    polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    // OSX intel + CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
#endif


     // boundary int and m1 index uint are known to be within char/uchar ranges 
    //  uchar: 0 to 255,   char: -128 to 127 

    qquad qaux ;  
    qaux.uchar_.x =  s_index.x ;    // m1  
    qaux.uchar_.y =  s_index.y ;    // m2   
    qaux.char_.z  =  p.flags.i.x ;  // boundary(range -55:55)   debugging some funny material codes
    qaux.uchar_.w = __ffs(s_flag) ; // first set bit __ffs(0) = 0, otherwise 1->32 
  
    //             lsb_ (flq[0].x)    msb_ (flq[0].y)
    //            
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;   
 
    //              lsb_ (flq[0].z)    msb_ (flq[0].w)
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;


    rbuffer[record_offset+1] = polw.short_ ; 
}


