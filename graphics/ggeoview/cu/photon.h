#pragma once

#define PNUMQUAD 4  // quads per photon  
#define RNUMQUAD 2  // quads per record  

// http://stackoverflow.com/questions/7337526/how-to-tell-if-a-32-bit-int-can-fit-in-a-16-bit-short
// see npy-/numutil.cpp
#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)

struct Photon
{
   float3 position ;  
   float  time ;  

   float3 direction ;
   float  wavelength ; 

   float3 polarization ;
   float  weight ; 

   quad flags ; 

};

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


__device__ void pinit( Photon& p)
{
   p.flags.i.x = 0 ;   
   p.flags.i.y = 0 ;   
   p.flags.i.z = 0 ;   
   p.flags.i.w = 0 ;   
}


__device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset )
{
    pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time ); 
    pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.wavelength );
    pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.weight );
    pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w); 
}


__device__ short shortnorm( float v, float center, float extent )
{
    // range of short is -32768 to 32767
    //
    // TODO: handle out of rangers (observe a few -ve times presumably due to this)
    //
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but out of time range eg 0.:100 ns is inevitable.
    //
    // http://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers

    float vnorm = 32767.0f * (v - center)/extent ;    // linear scaling into -1.f:1.f 

    int inorm = __float2int_rn(vnorm) ;  // 

    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;


    //  The below chapter, has a number of errors : not up to the 
    //  general high standard of that book
    //      http://www.informit.com/articles/article.aspx?p=2033340&seqNum=3
    //
    //   f = c / (2^b - 1)          for signed -1:1 case
    //   f = (2c + 1)/( 2^b - 1)    for unsigned 0:1 
    //
    //        (1 << 16) - 1 = 65535
    //
    //  Huh seems wrong ? should be 2^(b-1) ?  
    //
    //  http://stereopsis.com/radix.html
} 

__device__ void rsave( Photon& p, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
{
    // pack position and time into normalized shorts (16 bits)
    rbuffer[record_offset+0] = make_short4( 
                    shortnorm(p.position.x, center_extent.x, center_extent.w), 
                    shortnorm(p.position.y, center_extent.y, center_extent.w), 
                    shortnorm(p.position.z, center_extent.z, center_extent.w),   
                    shortnorm(p.time      , time_domain.x  , time_domain.y  )
                    ); 

    // TODO: rearrange bitfields to put more important ones into the lower 16 bits
    //       little-endian : LSB at smallest address ?
    //
    rbuffer[record_offset+1] = make_short4( p.flags.i.x & 0xFFFF , p.flags.i.y & 0xFFFF, p.flags.i.z, p.flags.i.w); 
}



