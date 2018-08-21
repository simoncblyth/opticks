#include "quad.h"
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void minimal()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*6 ; 

    union quad ipmn;
    ipmn.f = output_buffer[photon_offset+0];

    rtPrintf("RT %d\n", ipmn.i.w);

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();

    //unsigned long long photon_id = launch_index.x ;  
    //unsigned int photon_offset = photon_id*6 ; 
}



