#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#include "quad.h"

//#include "wavelength_lookup.h"


rtTextureSampler<float, 2>  reemission_texture ;
rtDeclareVariable(float4, reemission_domain, , );

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



RT_PROGRAM void textureTest()
{
   reemission_check();
}


RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



