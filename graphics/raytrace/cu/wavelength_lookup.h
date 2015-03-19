
#define NM_BLUE   475.f
#define NM_GREEN  510.f
#define NM_RED    650.f

rtTextureSampler<float4, 2>  wavelength_texture ;
rtDeclareVariable(float3, wavelength_domain, , );

static __device__ __inline__ float4 wavelength_lookup(float nm, float line )
{
    // x:low y:high z:step    tex coords are offset by 0.5 
    float nmi = (nm - wavelength_domain.x)/wavelength_domain.z + 0.5 ;   
    return tex2D(wavelength_texture, nmi, line );
}

static __device__ __inline__ void wavelength_dump(float line )
{
  for(int i=-5 ; i < 45 ; i++ )
  { 
     float nm = wavelength_domain.x + wavelength_domain.z*i ; 
     float4 lookup = wavelength_lookup( nm, line ); 
     rtPrintf("wavelength_dump i %2d nm %10.3f line %10.3f  lookup  %10.3f %10.3f %10.3f %10.3f \n", 
        i,
        nm,
        line,
        lookup.x,
        lookup.y,
        lookup.z,
        lookup.w);
  } 
}



