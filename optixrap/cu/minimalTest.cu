#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4,2>  output_buffer;

RT_PROGRAM void minimal()
{
    output_buffer[launch_index] = make_float4(42.f, 42.f, 42.f, 42.f);
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
    output_buffer[launch_index] = make_float4(-42.f, -42.f, -42.f, -42.f);
}



