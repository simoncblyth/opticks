#pragma once

struct State 
{
   float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
   float4 material2 ;  
   float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
   float3 surface_normal ; 
   float cos_theta ; 
   float distance_to_boundary ;
};


__device__ void fill_state( State& s, int boundary, float wavelength )
{
    // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
    // >0 outward going photon
    // <0 inward going photon

    int line = boundary > 0 ? (boundary - 1)*6 : (-boundary - 1)*6  ; 
    s.material1 = wavelength_lookup( wavelength, boundary > 0 ? line + 0 : line + 1 );  // inner-material / outer-material
    s.material2 = wavelength_lookup( wavelength, boundary > 0 ? line + 1 : line + 0 );  // outer-material / inner-material
    s.surface   = wavelength_lookup( wavelength, boundary > 0 ? line + 2 : line + 3 );  // inner-surface  / outer-surface
}

__device__ void  dump_state( State& s)
{
    rtPrintf(" dump_state:material1  %10.4f %10.4f %10.4f %10.4f \n", s.material1.x, s.material1.y, s.material1.z, s.material1.w );
    rtPrintf(" dump_state:material2  %10.4f %10.4f %10.4f %10.4f \n", s.material2.x, s.material2.y, s.material2.z, s.material2.w );
    rtPrintf(" dump_state:surface    %10.4f %10.4f %10.4f %10.4f \n", s.surface.x, s.surface.y, s.surface.z, s.surface.w );
}


