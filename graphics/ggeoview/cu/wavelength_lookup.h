#ifndef WAVELENGTH_LOOKUP_H 
#define WAVELENGTH_LOOKUP_H 

#define NM_BLUE   475.f
#define NM_GREEN  510.f
#define NM_RED    650.f

rtTextureSampler<float4, 2>  wavelength_texture ;
rtDeclareVariable(float3, wavelength_domain, , );

static __device__ __inline__ float4 wavelength_lookup(float nm, unsigned int line )
{
    // x:low y:high z:step    tex coords are offset by 0.5 
    float nmi = (nm - wavelength_domain.x)/wavelength_domain.z + 0.5 ;   
    return tex2D(wavelength_texture, nmi, line + 0.5f );
}

static __device__ __inline__ void wavelength_dump(unsigned int line )
{
  for(int i=-5 ; i < 45 ; i++ )
  { 
     float nm = wavelength_domain.x + wavelength_domain.z*i ; 
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
     unsigned int line = isub*4 + jqwn ; 
     float4 props = wavelength_lookup( wavelength, line ) ;
     rtPrintf("wavelength_check %10.3f nm isub %2u jqwn %u line %3u  props  %13.4f %13.4f %13.4f %13.4f \n",
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

#endif
