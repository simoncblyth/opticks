#include <optix_world.h>

using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

rtBuffer<float,3> OSensorLibTest_out ;

#include "OSensorLib.hh" 

/**
NB buffer/texture/launch shapes are transposed relative to the NPY array::
 
   array                 :  (num_cat, num_theta, num_phi)
   buffer/texture/launch :  (num_phi, num_theta, num_cat)
   
This is a done to reconcile the row-major serialization order used by NPY arrays(following NumPy) 
and the column-major serialization order used by OptiX buffers, see tests/OCtx2dTest.cc tests/OCtx3dTest.cc 
**/

RT_PROGRAM void raygen()
{
/*
    rtPrintf("//raygen launch_index (%d %d %d) launch_dim (%d %d %d) \n",
        launch_index.x, launch_index.y, launch_index.z,
        launch_dim.x, launch_dim.y, launch_dim.z );
*/

    int   category = int(launch_index.z) ;  
    float phi_fraction = (float(launch_index.x)+0.5f)/float(launch_dim.x) ; 
    float theta_fraction = (float(launch_index.y)+0.5f)/float(launch_dim.y) ; 

    float angular_efficiency = OSensorLib_angular_efficiency( category, phi_fraction , theta_fraction ); 

    OSensorLibTest_out[launch_index] = angular_efficiency ; 
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


