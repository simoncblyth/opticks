#include <optix_world.h>
using namespace optix;

rtBuffer<float4, 1>  output_buffer;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );


RT_PROGRAM void circle_make_vertices()
{
    float frac = float(launch_index)/float(launch_dim) ; 
    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*frac,&sinPhi,&cosPhi);

    if(launch_index < 10)
        rtPrintf( "circle_make_vertices launch dim %d index %d frac %10.4f s %10.4f c %10.4f \n", launch_dim, launch_index, frac, sinPhi, cosPhi);

    output_buffer[launch_index] = make_float4( sinPhi,  cosPhi,  0.0f, 1.0f) ;
}


RT_PROGRAM void circle_dump()
{
    float4 v = output_buffer[launch_index] ; 
    if(launch_index < 10)
        rtPrintf( "circle_dump (dim,index) (%d,%d) [%10.4f,%10.4f,%10.4f,%10.4f] \n", 
            launch_dim, launch_index, v.x, v.y, v.z, v.w  );
}



