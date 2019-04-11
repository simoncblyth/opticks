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
    //rtPrintf("bufferTest.cu:bufferTest  llu %llu  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
    rtPrintf("bufferTest.cu:bufferTest  d %d  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
    photon_buffer[index] = val ; 
}


RT_PROGRAM void bufferTest_0()
{
    unsigned long long index = launch_index.x ;
    rtPrintf("bufferTest.cu:bufferTest_0  llu : %llu \n", index );
}

RT_PROGRAM void bufferTest_1()
{
    unsigned long long index = launch_index.x ;
    float4 val = genstep_buffer[index] ; 
    rtPrintf("bufferTest.cu:bufferTest_1  llu : %llu  gs (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
}

RT_PROGRAM void bufferTest_2()
{
    unsigned long long index = launch_index.x ;
    float4 val = photon_buffer[index] ; 
    rtPrintf("bufferTest.cu:bufferTest_2  llu : %llu  ph (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
}





RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



