#pragma once

#include "define.h"

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



static __device__ __inline__ float4 wavelength_lookup(float nm, unsigned int line )
{
    // x:low y:high z:step w:mid   tex coords are offset by 0.5 
    // texture lookups benefit from hardware interpolation 
    float nmi = (nm - boundary_domain.x)/boundary_domain.z + 0.5f ;   

    if( line > boundary_bounds.w )
    {
        rtPrintf("wavelength_lookup OUT OF BOUNDS line %4d nmi %10.4f \n", line, nmi );
    }

    return line <= boundary_bounds.w ? 
                  tex2D(boundary_texture, nmi, line + 0.5f ) : 
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
     float4 lookup = wavelength_lookup( nm, line ); 
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
  float wavelength = NM_GREEN ;  
  for(unsigned int isub=0 ; isub < 100 ; ++isub)
  {
  for(unsigned int jqwn=0 ; jqwn < 1 ; ++jqwn)
  { 
     unsigned int line = isub*BNUMQUAD + jqwn ; 
     float4 props = wavelength_lookup( wavelength, line ) ;
     rtPrintf("wavelength_check BNUMQUAD %10.3f nm isub %2u jqwn %u line %3u  props  %13.4f %13.4f %13.4f %13.4f \n",
          wavelength,
          isub,
          jqwn, 
          line,
          props.x, 
          props.y, 
          props.z, 
          props.w
     ); 
  }

  }
}

