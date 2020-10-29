#include <optix_device.h>

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<int4,2> buffer;

RT_PROGRAM void raygen()
{
    buffer[launch_index] = make_int4(launch_index.x, launch_index.y, launch_dim.x, launch_dim.y ) ; 
}
RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


