#pragma once

#define NM_BLUE   475.f
#define NM_GREEN  510.f
#define NM_RED    650.f

rtTextureSampler<float, 2>  source_texture ;
rtDeclareVariable(float4, source_domain, , );

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
#ifdef WITH_PRINT
    float nm_a = source_lookup(0.0f); 
    float nm_b = source_lookup(0.5f); 
    float nm_c = source_lookup(1.0f); 
    rtPrintf("source_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
#endif
}

/*
source_check nm_a     60.000    506.041    820.000  
source_check nm_a     60.000    506.041    820.000  
*/


