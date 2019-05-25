#include <optix_world.h>

using namespace optix;

rtTextureSampler<float4, 2>  boundary_texture ;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4,2>       out_buffer;


RT_PROGRAM void boundaryTest()
{
    int ix = int(launch_index.x) ; 
    int iy = int(launch_index.y) ; 

    float x = (float(ix)+0.5f)/float(launch_dim.x) ; 
    float y = (float(iy)+0.5f)/float(launch_dim.y) ; 
    
    float4 val = tex2D(boundary_texture, x, y );

    //rtPrintf("boundaryTest (%d,%d) (%10.4f,%10.4f) -> (%10.4f,%10.4f,%10.4f,%10.4f)  \n", ix, iy, x, y, val.x, val.y, val.z, val.w);

    out_buffer[launch_index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



