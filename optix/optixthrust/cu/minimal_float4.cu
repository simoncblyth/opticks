#include <optix_world.h>
using namespace optix;

rtBuffer<float4, 1>  output_buffer;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );


RT_PROGRAM void minimal()
{
    if(launch_index < 10) 
       rtPrintf( "minimal launch dim (%d) index (%d)\n", launch_dim, launch_index  );

    output_buffer[launch_index] = optix::make_float4( 0.f, 1.f, 2.f, 3.f );    
}

RT_PROGRAM void dump()
{
    float4 v = output_buffer[launch_index] ; 

    if(launch_index < 10) 
       rtPrintf( "dump (dim,index) (%d,%d) [%10.4f,%10.4f,%10.4f,%10.4f] \n", 
           launch_dim, launch_index, v.x, v.y, v.z, v.w  );
}


union fui {
  float        f ;
  unsigned int u ;
  int          i ;
};
  

RT_PROGRAM void circle()
{
    float frac = float(launch_index)/float(launch_dim) ; 
    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*frac,&sinPhi,&cosPhi);

    if(launch_index < 10) 
        rtPrintf( "circle launch dim %d index %d frac %10.4f s %10.4f c %10.4f \n", launch_dim, launch_index, frac, sinPhi, cosPhi);

    fui w ;
    w.u = launch_index % 10 ;    // tag the float4 

    output_buffer[launch_index] = make_float4( sinPhi, cosPhi,  0.0f, w.f) ;
}





