#include <optix_world.h>
//#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void minimal()
{
    unsigned photon_id = launch_index.x ;  
    unsigned photon_offset = photon_id*4 ; 
 
    rtPrintf("// redirectLogTest.cu:minimal %d \n", photon_id );
   
    output_buffer[photon_offset+0] = make_float4(40.f, 40.f, 40.f, 40.f);
    output_buffer[photon_offset+1] = make_float4(41.f, 41.f, 41.f, 41.f);
    output_buffer[photon_offset+2] = make_float4(42.f, 42.f, 42.f, 42.f);
    output_buffer[photon_offset+3] = make_float4(43.f, 43.f, 43.f, 43.f);

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



