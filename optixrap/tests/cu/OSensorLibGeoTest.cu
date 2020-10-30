#include <optix_world.h>

using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

//rtBuffer<float,3> OSensorLibTest_out ;

#include "OSensorLib.hh" 

RT_PROGRAM void raygen()
{
    /*
    int   category = int(launch_index.z) ;  
    float phi_fraction = (float(launch_index.x)+0.5f)/float(launch_dim.x) ; 
    float theta_fraction = (float(launch_index.y)+0.5f)/float(launch_dim.y) ; 

    float angular_efficiency = OSensorLib_angular_efficiency( category, phi_fraction , theta_fraction ); 

    OSensorLibTest_out[launch_index] = angular_efficiency ; 
    */
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


