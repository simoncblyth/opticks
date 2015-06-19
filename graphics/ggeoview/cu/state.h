#pragma once

rtBuffer<uint4>     optical_buffer; 

struct State 
{
   float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
   float4 material2 ;  
   float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
   float3 surface_normal ; 
   float cos_theta ; 
   float distance_to_boundary ;
   uint4 optical ;   // x/y/z/w index/type/finish/value  
   uint4 index ;     // indices of m1/m2/surf/sensor

};


__device__ void fill_state( State& s, int boundary, int sensor, float wavelength )
{
    // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
    // >0 outward going photon
    // <0 inward going photon

    int line = boundary > 0 ? (boundary - 1)*6 : (-boundary - 1)*6  ; 

    // pick relevant lines depening on boundary sign, ie photon direction relative to normal
    // 
    int m1 = boundary > 0 ? line + 0 : line + 1 ;   // inner-material / outer-material
    int m2 = boundary > 0 ? line + 1 : line + 0 ;   // outer-material / inner-material
    int su = boundary > 0 ? line + 2 : line + 3 ;   // inner-surface  / outer-surface

    s.material1 = wavelength_lookup( wavelength, m1 );  
    s.material2 = wavelength_lookup( wavelength, m2 ) ;
    s.surface   = wavelength_lookup( wavelength, su );                 

    s.optical = optical_buffer[su] ;   // index/type/finish/value

    s.index.x = optical_buffer[m1].x ;
    s.index.y = optical_buffer[m2].x ;
    s.index.z = optical_buffer[su].x ;
    s.index.w = sensor  ;

}

__device__ void  dump_state( State& s)
{
    rtPrintf(" dump_state:material1       %10.4f %10.4f %10.4f %10.4f \n", s.material1.x, s.material1.y, s.material1.z, s.material1.w );
    rtPrintf(" dump_state:material2       %10.4f %10.4f %10.4f %10.4f \n", s.material2.x, s.material2.y, s.material2.z, s.material2.w );
    rtPrintf(" dump_state:surface         %10.4f %10.4f %10.4f %10.4f \n", s.surface.x, s.surface.y, s.surface.z, s.surface.w );
    rtPrintf(" dump_state:optical         %10u %10u %10u %10i             \n", s.optical.x, s.optical.y, s.optical.z, s.optical.w );
    rtPrintf(" dump_state:index           %10u %10u %10u %10i  m1/m2/su/se \n", s.index.x  , s.index.y,   s.index.z,   s.index.w );
}


