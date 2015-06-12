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

   quad flags ; 

};

//
// flipped wavelength/weight as that puts together the quad that will be dropped for records
//  TODO: move to float4 eg position_time 
//

enum
{
    NO_HIT                 = 0x1 << 0,
    BULK_ABSORB            = 0x1 << 1,
    SURFACE_DETECT         = 0x1 << 2,
    SURFACE_ABSORB         = 0x1 << 3,
    RAYLEIGH_SCATTER       = 0x1 << 4,
    REFLECT_DIFFUSE        = 0x1 << 5,
    REFLECT_SPECULAR       = 0x1 << 6,
    SURFACE_REEMIT         = 0x1 << 7,
    SURFACE_TRANSMIT       = 0x1 << 8,
    BULK_REEMIT            = 0x1 << 9,
    GENERATE_SCINTILLATION = 0x1 << 16, 
    GENERATE_CERENKOV      = 0x1 << 17, 
    BOUNDARY_SPOL          = 0x1 << 18,
    BOUNDARY_PPOL          = 0x1 << 19,
    BOUNDARY_REFLECT       = 0x1 << 20,
    BOUNDARY_TRANSMIT      = 0x1 << 21,
    BOUNDARY_TIR           = 0x1 << 22,
    BOUNDARY_TIR_NOT       = 0x1 << 23,
    NAN_ABORT              = 0x1 << 31
}; // processes




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
    //  range of char is -128 to 127, normalization of polarization and wavelength expected bulletproof, so no handling of out-of-range 
    //
    //  polarization already normalized into -1.f:1.f
    //  wavelenth normalized via  (wavelength - low)/range into 0.:1. 
    //

    float nwavelength = 255.f*(p.wavelength - wavelength_domain.x)/wavelength_domain.w ; // 255.f*0.f->1.f 


    // lightly packed 
    /*
    rbuffer[record_offset+1] = make_short4( 
                                __float2int_rn(p.polarization.x*127.f), 
                                __float2int_rn(p.polarization.y*127.f),
                                __float2int_rn(p.polarization.z*127.f),
                                __float2int_rn(nwavelength*127.f)
                              );
    */


    // range of uchar 0:255   -1.f:1.f  + 1 => 0.f:2.f  so scale by 127.f 
    qquad flags ;    
    flags.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
    flags.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    flags.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    flags.uchar_.w = __float2uint_rn(nwavelength)  ;

    // tightly packed, 
    hquad polw ; 
    polw.ushort_.x = flags.uchar_.x | flags.uchar_.y << 8 ;
    polw.ushort_.y = flags.uchar_.z | flags.uchar_.w << 8 ;
    
    // maps to rflg.x rflg.y in shader

#ifdef IDENTITY_CHECK
    // spread uint32 photon_id across two uint16
    unsigned int photon_id = p.flags.u.y ;
    polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    // OSX intel, CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
#endif

    // range of 8 bit char: -128 to 127 or 0 to 255
    int boundary = p.flags.i.x ;  // range -55 : 55 
    //polw.ushort_.z = (boundary & 0xFF) << 8 ;
    polw.ushort_.z = boundary  ;



    rbuffer[record_offset+1] = polw.short_ ; 
}

/*

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



