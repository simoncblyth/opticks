
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;


rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(int4,  tex_param, , );
rtBuffer<float,2>           out_buffer;

// contrast with texTest which uses float4 access to the texture



RT_PROGRAM void tex0Test()
{
    // texture indexing for (nx, ny)
    //      array type indexing:   0:nx-1 , 0:ny-1
    //      norm float indexing:   0:1-1/nx  , 0:1-1/ny

    int ix = int(launch_index.x) ; 
    int iy = int(launch_index.y) ; 

    float x = (float(ix)+0.5f)/float(launch_dim.x) ; 
    float y = (float(iy)+0.5f)/float(launch_dim.y) ; 
    
    int tex_id = tex_param.x ; 
    float val = rtTex2D<float>( tex_id, x, y ); 

    //rtPrintf("tex0Test (%d,%d) (%10.4f,%10.4f) -> %10.4f  \n", ix, iy, x, y, val);

    out_buffer[launch_index] = val ; 

}

RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



