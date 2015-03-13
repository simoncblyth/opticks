#include "materials.h"

// shadingNormal is set by the closest hit intersection program 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(unsigned int, node_index, attribute node_index, );

rtDeclareVariable(float3, contrast_color, , );
rtTextureSampler<float4, 2>  wavelength_texture ;
rtDeclareVariable(float3, wavelength_domain, , );


rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_touch, prd_touch, rtPayload, );

//
// 
RT_PROGRAM void closest_hit_radiance()
{
  //prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
  //prd_radiance.result = make_float3(0.f);
  prd_radiance.result = contrast_color ; 
}


static __device__ __inline__ float4 wlookup(float wavelength, float line )
{
    // x:low y:high z:step    tex coords are offset by 0.5 
    float wi = (wavelength - wavelength_domain.x)/wavelength_domain.z + 0.5 ;   
    return tex2D(wavelength_texture, wi, line );
}


RT_PROGRAM void closest_hit_touch()
{
  prd_touch.result = contrast_color ; 
  prd_touch.node = node_index ; 

  prd_touch.texlookup_b = wlookup( NM_BLUE  , 0.5f ) ;
  prd_touch.texlookup_g = wlookup( NM_GREEN , 0.5f ) ;
  prd_touch.texlookup_r = wlookup( NM_RED   , 0.5f ) ;

  for(int i=-5 ; i < 45 ; i++ )
  { 
     float wl = wavelength_domain.x + wavelength_domain.z*i ; 
     float4 lookup = wlookup( wl, 0.5f ); 
     rtPrintf("material1.cu::closest_hit_touch node %d   i %2d wl %10.3f   lookup  %10.3f %10.3f %10.3f %10.3f \n", 
        node_index,
        i,
        wl,
        lookup.x,
        lookup.y,
        lookup.z,
        lookup.w);
  } 
}

