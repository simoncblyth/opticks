#include <optix_world.h>
using namespace optix;

rtBuffer<float,1> result_buffer ; 

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

RT_PROGRAM void readOnly()
{
    float val = 42.f ; 
    rtPrintf("//UseOptiXBuffer:readOnly launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u  val %10.3f \n", launch_index.x , launch_index.y, launch_dim.x , launch_dim.y, val    );
    result_buffer[launch_index.x] = val  ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


