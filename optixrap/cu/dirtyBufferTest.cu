#include <optix_world.h>

using namespace optix;

rtBuffer<unsigned>  test_buffer ;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


RT_PROGRAM void dirtyBufferTest()
{
    unsigned long long index = launch_index.x ;
    unsigned val = test_buffer[index] ; 
    rtPrintf("dirtyBufferTest.cu  %u )  \n", val );
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



