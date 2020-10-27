#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(int,  tex_id, , );

rtBuffer<float,2> output_buffer;


RT_PROGRAM void raygen()
{
    // texture indexing for (nx, ny)
    //      array type indexing:   0:nx-1 , 0:ny-1
    //      norm float indexing:   0:1-1/nx  , 0:1-1/ny

    int ix = int(launch_index.x) ; 
    int iy = int(launch_index.y) ; 

    float x = (float(ix)+0.5f)/float(launch_dim.x) ; // width:phi
    float y = (float(iy)+0.5f)/float(launch_dim.y) ; // height:theta 
    
    float val = rtTex2D<float>( tex_id, x, y ); 

    output_buffer[launch_index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



