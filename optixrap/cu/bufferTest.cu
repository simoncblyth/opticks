#include <optix_world.h>

using namespace optix;

rtBuffer<float4>  genstep_buffer ;
rtBuffer<float4>  photon_buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


RT_PROGRAM void bufferTest()
{
    unsigned long long index = launch_index.x ;
    float4 val = genstep_buffer[index] ; 
//    rtPrintf("bufferTest.cu  %d  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
    photon_buffer[index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



