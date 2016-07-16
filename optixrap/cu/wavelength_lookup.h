#pragma once

#include "define.h"
#include "GPropertyLib.hh"


#define NM_BLUE   475.f
#define NM_GREEN  510.f
#define NM_RED    650.f

rtTextureSampler<float, 2>  reemission_texture ;
rtDeclareVariable(float4, reemission_domain, , );

rtTextureSampler<float, 2>  source_texture ;
rtDeclareVariable(float4, source_domain, , );


rtTextureSampler<float4, 2>  boundary_texture ;
rtDeclareVariable(float4, boundary_domain, , );
rtDeclareVariable(float4, boundary_domain_reciprocal, , );
rtDeclareVariable(uint4, boundary_bounds, , );


static __device__ __inline__ float reemission_lookup(float u)
{
    float ui = u/reemission_domain.z + 0.5f ;   
    return tex2D(reemission_texture, ui, 0.5f );  // line 0
}

static __device__ __inline__ void reemission_check()
{
    float nm_a = reemission_lookup(0.0f); 
    float nm_b = reemission_lookup(0.5f); 
    float nm_c = reemission_lookup(1.0f); 
    rtPrintf("reemission_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
}

static __device__ __inline__ float comb_lookup(float u)
{
     float nm(400.0f) ; 

     if(     u < 0.20f) nm = 400.0f ;
     else if(u < 0.40f) nm = 500.0f ; 
     else if(u < 0.60f) nm = 600.0f ; 
     else if(u < 0.80f) nm = 700.0f ; 
     else               nm = 800.0f ; 

     return nm ; 
}


static __device__ __inline__ float source_lookup(float u)
{
    float ui = u/source_domain.z + 0.5f ;   
    return tex2D(source_texture, ui, 0.5f );  // line 0
}

static __device__ __inline__ void source_check()
{
    float nm_a = source_lookup(0.0f); 
    float nm_b = source_lookup(0.5f); 
    float nm_c = source_lookup(1.0f); 
    rtPrintf("source_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
}

/*
source_check nm_a     60.000    506.041    820.000  
source_check nm_a     60.000    506.041    820.000  
*/



static __device__ __inline__ float4 wavelength_lookup(float nm, unsigned int line, unsigned int offset )
{
    // x:low y:high z:step w:mid   tex coords are offset by 0.5 
    // texture lookups benefit from hardware interpolation 
    float nmi = (nm - boundary_domain.x)/boundary_domain.z + 0.5f ;   

    if( line > boundary_bounds.w )
    {
        rtPrintf("wavelength_lookup OUT OF BOUNDS nm %10.4f nmi %10.4f line %4d offset %4d boundary_bounds (%4u,%4u,%4u,%4u) boundary_domain (%10.4f,%10.4f,%10.4f,%10.4f) \n", 
            nm,
            nmi,
            line,
            offset,
            boundary_bounds.x,
            boundary_bounds.y,
            boundary_bounds.z,
            boundary_bounds.w,
            boundary_domain.x,
            boundary_domain.y,
            boundary_domain.z,
            boundary_domain.w);
    }

    return line <= boundary_bounds.w ? 
                  tex2D(boundary_texture, nmi, BOUNDARY_NUM_FLOAT4*line + offset + 0.5f ) : 
                  make_float4(1.123456789f, 123456789.f, 123456789.f, 1.0f )    ;    // some obnoxious values for debug 

    // refractive_index, absorption_length, scattering_length, reemission_prob
    // DEBUG KLUDGE
}

static __device__ __inline__ float sample_reciprocal_domain(const float& u)
{
    // return wavelength, from uniform sampling of 1/wavelength[::-1] domain
    float iw = lerp( boundary_domain_reciprocal.x , boundary_domain_reciprocal.y, u ) ;
    return 1.f/iw ;  
}

static __device__ __inline__ float sample_domain(const float& u)
{
    // return wavelength, from uniform sampling of wavelength domain
    return lerp( boundary_domain.x , boundary_domain.y, u ) ;
}



static __device__ __inline__ void wavelength_dump(unsigned int line, unsigned int step)
{
  for(int i=-5 ; i < 45 ; i+=step )
  { 
     float nm = boundary_domain.x + boundary_domain.z*i ; 
     float4 lookup = wavelength_lookup( nm, line, 0 ); 
     rtPrintf("wavelength_dump i %2d nm %10.3f line %u  lookup  %10.3f %10.3f %10.3f %10.3f \n", 
        i,
        nm,
        line,
        lookup.x,
        lookup.y,
        lookup.z,
        lookup.w);
  } 
}



static __device__ __inline__ void wavelength_check()
{
  unsigned int ibnd=13 ; 

  unsigned int jqwn=0 ;  //OMAT
 
  for(int i=0 ; i < 39 ; i++ )
  {
     float nm = boundary_domain.x + boundary_domain.z*i ; 

     unsigned int line = ibnd*BOUNDARY_NUM_MATSUR + jqwn ; 

     float4 pr0 = wavelength_lookup( nm, line, 0 ) ;
     float4 pr1 = wavelength_lookup( nm, line, 1 ) ;

     rtPrintf("wavelength_check nm:%10.3f ibnd %2u jqwn %u line %3u  pr0  %13.4f %13.4f %13.4f %13.4f pr1  %13.4f %13.4f %13.4f %13.4f \n",
          nm, 
          ibnd,
          jqwn, 
          line,
          pr0.x, 
          pr0.y, 
          pr0.z, 
          pr0.w,
          pr1.x, 
          pr1.y, 
          pr1.z, 
          pr1.w
     ); 
  }
}

