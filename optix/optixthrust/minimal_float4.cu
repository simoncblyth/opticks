#include <optix_world.h>
using namespace optix;

rtBuffer<float4, 1>  output_buffer;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );


RT_PROGRAM void minimal_float4()
{
    rtPrintf( "minimal_float4 launch dim (%d) index (%d)\n", launch_dim, launch_index  );
    output_buffer[launch_index] = optix::make_float4( 0.f, 1.f, 2.f, 3.f );    
}




