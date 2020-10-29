#include <optix_device.h>

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

rtBuffer<int4,3> buffer;

RT_PROGRAM void raygen()
{
    buffer[launch_index] = make_int4(launch_index.x, launch_index.y, launch_index.z, launch_dim.z ) ; 
}
RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


