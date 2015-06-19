#pragma once

// see npy-/numutil.cpp
// http://stackoverflow.com/questions/7337526/how-to-tell-if-a-32-bit-int-can-fit-in-a-16-bit-short
// http://en.wikipedia.org/wiki/Two's_complement
// http://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers
// http://stereopsis.com/radix.html
// https://graphics.stanford.edu/~seander/bithacks.html
// http://www.informit.com/articles/article.aspx?p=2033340&seqNum=3

#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)


struct Photon
{
   float3 position ;  
   float  time ;  

   float3 direction ;
   float  weight ; 

   float3 polarization ;
   float  wavelength ; 

   quad flags ;     // x:boundary  y:photon_id   z:spare   w:history 
                    //             [debug-only]
};


/*

In [5]: p.view(np.int32)[:,3]
Out[5]: 
array([[   -14,      0,      0, 526857],
       [     0,      1,      0, 531973],
       [   -12,      2,      0,      9],
       ..., 
       [     0, 612838,      0,      5],
       [     0, 612839,      0,      5],
       [     0, 612840,      0,      5]], dtype=int32)

*/



//
// flipped wavelength/weight as that puts together the quad that will be dropped for records
//  TODO: move to float4 eg position_time 
//

enum
{
    GENERATE_CERENKOV      = 0x1 << 0, 
    GENERATE_SCINTILLATION = 0x1 << 1, 
    NO_HIT                 = 0x1 << 2,
    BULK_ABSORB            = 0x1 << 3,
    SURFACE_DETECT         = 0x1 << 4,
    SURFACE_ABSORB         = 0x1 << 5,
    RAYLEIGH_SCATTER       = 0x1 << 6,
    BULK_REEMIT            = 0x1 << 7,
    BOUNDARY_SPOL          = 0x1 << 8,
    BOUNDARY_PPOL          = 0x1 << 9,
    BOUNDARY_REFLECT       = 0x1 << 10,
    BOUNDARY_TRANSMIT      = 0x1 << 11,
    BOUNDARY_TIR           = 0x1 << 12,
    NAN_ABORT              = 0x1 << 13,
    REFLECT_DIFFUSE        = 0x1 << 14, 
    REFLECT_SPECULAR       = 0x1 << 15,
    SURFACE_REEMIT         = 0x1 << 17,
    SURFACE_TRANSMIT       = 0x1 << 18,
    BOUNDARY_TIR_NOT       = 0x1 << 19
}; // processes

//  only 0-15 make it into the record so debug flags only beyond 15 


enum { BREAK, CONTINUE, PASS, START, RETURN }; // return value from propagate_to_boundary


__device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset )
{
    pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time ); 
    pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
    pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
    pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w); 
}


__device__ short shortnorm( float v, float center, float extent )
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //
    int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 


__device__ void rsave( Photon& p, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
{
    //  pack position and time into normalized shorts (4*16 = 64 bits)
    //
    //  TODO: use a more vectorized approach, ie
    // 
    //  * combine position and time domains into single float4 on the host 
    //  * after verification can dispense with the fit checking for positions, just do time
    //        
    //  * adopt p.position_time  maybe p.polarization_wavelength
    //  * simularly with domains of those ?
    // 
    rbuffer[record_offset+0] = make_short4( 
                    shortnorm(p.position.x, center_extent.x, center_extent.w), 
                    shortnorm(p.position.y, center_extent.y, center_extent.w), 
                    shortnorm(p.position.z, center_extent.z, center_extent.w),   
                    shortnorm(p.time      , time_domain.x  , time_domain.y  )
                    ); 

    //  pack polarization and wavelength into 4*8 = 32 bits   
    //  range of char -128:127  normalization of polarization and wavelength expected bulletproof, so no handling of out-of-range 
    //  range of uchar 0:255   -1.f:1.f  + 1 => 0.f:2.f  so scale by 127.f 
    //
    //  polarization already normalized into -1.f:1.f
    //  wavelenth normalized via  (wavelength - low)/range into 0.:1. 

    float nwavelength = 255.f*(p.wavelength - wavelength_domain.x)/wavelength_domain.w ; // 255.f*0.f->1.f 

    qquad qpolw ;    
    qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
    qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;

    hquad polw ; // tightly packed, polarization and wavelength  
    polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;
    

#ifdef IDENTITY_CHECK
    // spread uint32 photon_id across two uint16
    unsigned int photon_id = p.flags.u.y ;
    polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    // OSX intel, CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
#endif


    qquad qaux ;  // boundary int and m1 index uint are known to be within char/uchar ranges 
    qaux.char_.x  =  p.flags.i.x ;  //  boundary(range -55:55)    char: -128 to 127  
    qaux.uchar_.y =  p.flags.u.z ;  //  m1 index                 uchar: 0 to 255

    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8 ;
    polw.ushort_.w = p.flags.u.w & 0xFFFF  ;      // 16 bits of history 

    rbuffer[record_offset+1] = polw.short_ ; 

    // flags intended to allow tracing photon propagation history via bitfields for each record 
    // of a photon, each corresponding to bits set since the prior rsave (not the same as a step)
    // although exclusive or on a "|=" incrementing mask almost does this 
    // that would fail to see repeated flags  

}

/*

   (m1/bd check)

::

    In [3]: a = rxc_(1)


    In [55]: m1 = np.array( a[:,1,2] >> 8, dtype=np.int8 )    ## all records, 2nd quad, z, high char

    In [56]: m1
    Out[56]: array([  57,   57,   72, ..., -128, -128, -128], dtype=int8)

    In [57]: m1[:1000].reshape(-1,10)
    Out[57]: 
    array([[  57,   57,   72,   72, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57,   57, -128, -128, -128, -128, -128, -128],
           [  57,   57, -128, -128, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57,   57, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57,   57, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57, -128, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57, -128, -128, -128, -128, -128, -128, -128],
           [  57,   57,   57, -128, -128, -128, -128, -128, -128, -128],
           [  57,   57, -128, -128, -128, -128, -128, -128, -128, -128],
           [  57,   57, -128, -128, -128, -128, -128, -128, -128, -128],



    In [50]: bd = np.array( a[:,1,2] & 0xFF , dtype=np.int8 )

    In [51]: bd
    Out[51]: array([-12, -13, -14, ...,   0,   0,   0], dtype=int8)

    In [52]: bd[:1000].reshape(-1,10)
    Out[52]: 
    array([[-12, -13, -14, -14,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12, -12, -12,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],





   (identity check, depending on same endianness of host and device)

::

    In [1]: r = rxc_(1)

    In [2]: (r.shape, r.dtype)
    Out[2]: ((6128410, 2, 4), dtype('int16'))

    In [7]: r.view(np.uint16).view(np.uint32)[::10,1,1]   # scrunch up 2 uint16 into a uint32
    Out[7]: array([     0,      1,      2, ..., 612838, 612839, 612840], dtype=uint32)

    In [8]: np.all(np.arange(0,612841,dtype=np.uint32) == r.view(np.uint32)[::10,1,1])  ## chaining views dont make much sense
    Out[8]: True


    ## alternatively without the dirty scrunching 

    In [30]: lss = np.array( r.view(np.uint16)[::10,1,2], dtype=np.uint32 )  ## flags.z

    In [31]: mss = np.array( r.view(np.uint16)[::10,1,3], dtype=np.uint32 )  ## flags.w

    In [32]: mss << 16 | lss 
    Out[32]: array([     0,      1,      2, ..., 612838, 612839, 612840], dtype=uint32)

    In [33]: np.all( (mss << 16) | lss == np.arange(0,612841, dtype=np.uint32 ))
    Out[33]: True



    (lightly packed) 

    check packing results using ::MAXREC to pick generated 0th slot only  

    In [1]: a = rxc_(1)

       plt.hist(a[::10,1,0], range=(-128,127), bins=256 )  # bins+1 edges 
       plt.hist(a[::10,1,1], range=(-128,127), bins=256 )  # bins+1 edges 
       plt.hist(a[::10,1,2], range=(-128,127), bins=256 )  # bins+1 edges 
  
       plt.hist(a[::10,1,3], range=(-128,127), bins=256 )  # bins+1 edges 


    (tightly packed, using signed normalization)  

     hmm get no negatives, for tight packing better to use unsigned 

       plt.hist( a[::10,1,0] & 0xFF , bins=256, range=(-128,127) ) 


    (tightly packed, using unsigned normalization)

       plt.hist( a[::10,1,0] & 0xFF , bins=256, range=(0,256) )
       plt.hist( a[::10,1,0] >> 8   , bins=256, range=(0,256) )
       plt.hist( a[::10,1,1] & 0xFF , bins=256, range=(0,256) )

       plt.hist( a[::10,1,1] >> 8   , bins=256, range=(0,256) )  ## oops runs into sign bit yields halfed and folded

       plt.hist( a.view(np.uint16)[::10,1,1] >> 8, bins=256, range=(0,255) )   ## zero bin lower... due to bad number of bins
       plt.hist( a.view(np.uint16)[::10,1,1] >> 8, bins=255, range=(0,255) )   ## better mpl provides bins+1 edges
                                               
       wavelength is occupyung the MSB, so runs into the sign bit 
       hence its necessary to view(np.uint16)
       its more correct to do that for all when using unorm but only matters for MSB

 



*/



