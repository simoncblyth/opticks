#pragma once
#define PNUMQUAD 4  // quads per photon  

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
    NO_HIT           = 0x1 << 0,
    BULK_ABSORB      = 0x1 << 1,
    SURFACE_DETECT   = 0x1 << 2,
    SURFACE_ABSORB   = 0x1 << 3,
    RAYLEIGH_SCATTER = 0x1 << 4,
    REFLECT_DIFFUSE  = 0x1 << 5,
    REFLECT_SPECULAR = 0x1 << 6,
    SURFACE_REEMIT   = 0x1 << 7,
    SURFACE_TRANSMIT = 0x1 << 8,
    BULK_REEMIT      = 0x1 << 9,
    GENERATE_SCINTILLATION = 0x1 << 16, 
    GENERATE_CERENKOV      = 0x1 << 17, 
    NAN_ABORT        = 0x1 << 31
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



