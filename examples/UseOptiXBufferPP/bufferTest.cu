#include <optix_world.h>
using namespace optix;

rtBuffer<float4> in_buffer ; 
rtBuffer<float4> out_buffer ; 
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//#define WITH_PRINT 1

//#define WITH_EXCEPTION 1


RT_PROGRAM void bufferTest_readWrite()
{
    unsigned long long index = launch_index.x ;
    float4 val = in_buffer[index] ; 
#ifdef WITH_PRINT
    rtPrintf("//bufferTest llu:%llu x %10.3f y %10.3f z %10.3f w %10.3f \n", index, val.x, val.y, val.z, val.w   );
#endif
    out_buffer[index] = val ; 
}

RT_PROGRAM void bufferTest_readOnly()
{
     unsigned long long index = launch_index.x ;
    // float4 val = in_buffer[index] ; 
    // rtPrintf("//bufferTest llu:%llu x %10.3f y %10.3f z %10.3f w %10.3f \n", index, val.x, val.y, val.z, val.w   );
    // out_buffer[index] = val ; 
    out_buffer[index] = make_float4( 1.f, 2.f, 3.f, 4.f );   
}



RT_PROGRAM void printTest0()
{
     unsigned long long index = launch_index.x ;
     rtPrintf("//printTest0 d:%d launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", index, launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
}
RT_PROGRAM void printTest1()
{
     unsigned long long index = launch_index.x ;
     rtPrintf("//printTest1 llu:%llu launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", index, launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
}


RT_PROGRAM void exception()
{
#ifdef WITH_EXCEPTION
    rtPrintExceptionDetails();
#endif
}


