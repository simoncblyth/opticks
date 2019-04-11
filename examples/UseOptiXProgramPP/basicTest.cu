#include <optix_world.h>
using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


RT_PROGRAM void basicTest()
{
     rtPrintf("//basicTest launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
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
    rtPrintExceptionDetails();
}


