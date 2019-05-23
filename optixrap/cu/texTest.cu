
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;


rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(int4,  tex_param, , );

rtBuffer<float4,2>           out_buffer;

//#define WITH_PRINT 1


RT_PROGRAM void texTest()
{
    // texture indexing for (nx, ny)
    //      array type indexing:   0:nx-1 , 0:ny-1
    //      norm float indexing:   0:1-1/nx  , 0:1-1/ny

    int ix = int(launch_index.x) ; 
    int iy = int(launch_index.y) ; 

    float x = (float(ix)+0.5f)/float(launch_dim.x) ; 
    float y = (float(iy)+0.5f)/float(launch_dim.y) ; 
    
 //   float val = tex2D(some_texture, x, y );

    int tex_id = tex_param.x ; 
    float4 val = rtTex2D<float4>( tex_id, x, y ); 

#ifdef WITH_PRINT
    rtPrintf("texTest (%d,%d) (%10.4f,%10.4f) -> (%10.4f,%10.4f,%10.4f,%10.4f)  \n", ix, iy, x, y, val.x, val.y, val.z, val.w);
#endif

    out_buffer[launch_index] = val ; 

}

RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



