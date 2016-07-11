#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2,  launch_index, rtLaunchIndex, );
rtDeclareVariable(float4, center_extent, , );

rtBuffer<float4>  axis_buffer;

RT_PROGRAM void axisModify()
{
    // buffer shape is 3,3,4
    // but address it here as 9*4 

    unsigned long long axis_id = launch_index.x ;  // 0,1,2 for X,Y,Z axes 
    unsigned int   axis_offset = axis_id*3 ; 
    
    axis_buffer[axis_offset+0] = center_extent ; 

    if(axis_id == 0)
    {
         axis_buffer[axis_offset+1] = make_float4(0.f, 1000.f, 1000.f, 0.f);   
         axis_buffer[axis_offset+2] = make_float4(0.f, 0.f, 1.f, 1.f); // vcol
    }
    else if(axis_id == 1)
    {
         axis_buffer[axis_offset+1] = make_float4(1000.f, 0.f, 1000.f, 0.f); // vdir   
         axis_buffer[axis_offset+2] = make_float4(0.f, 1.f, 0.f, 1.f); // vcol
    }
    else if(axis_id == 2)
    {
         axis_buffer[axis_offset+1] = make_float4(1000.f, 1000.f, 0.f, 0.f); // vdir   
         axis_buffer[axis_offset+2] = make_float4(1.f, 0.f, 0.f, 1.f); // vcol
    }

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



