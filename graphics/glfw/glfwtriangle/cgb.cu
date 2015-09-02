#include <optix_world.h>
using namespace optix;

rtBuffer<float3, 1>  cgb_buffer;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );


RT_PROGRAM void cgb()
{
    rtPrintf( "cgb.cu launch dim %d index %d \n", launch_dim, launch_index  );
    
    float s = 0.5f ; 
    float3 vec ; 
    switch(launch_index)
    {
        case 0: vec = make_float3( 0.0f,     s,  0.0f) ; break ;
        case 1: vec = make_float3(    s,    -s,  0.0f) ; break ;
        case 2: vec = make_float3(   -s,    -s,  0.0f) ; break ;
    }  

    cgb_buffer[launch_index] = vec ;    
}




