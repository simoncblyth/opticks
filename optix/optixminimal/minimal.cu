#include <optix_world.h>
using namespace optix;

rtBuffer<uchar4, 2>  output_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


RT_PROGRAM void minimal()
{
    rtPrintf( "minimal launch dim (%d,%d) index (%d,%d)\n", launch_dim.x, launch_dim.y, launch_index.x, launch_index.y );
    output_buffer[launch_index] = optix::make_uchar4(128u, 128u, 128u, 128u );    
}




