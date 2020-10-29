#include <optix_world.h>

using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

rtBuffer<float,3> output_buffer;
rtBuffer<int4,1>    texid_buffer ; 

/**
Note that buffer (and launch) shapes are transposed compared to the array::
 
   Array shape          :  (num_cat, num_theta, num_phi)
   Buffer/Launch shape  :  (num_phi, num_theta, num_cat)
   
This is a done as a workaround to reconcile the row-major serialization
order used by NPY arrays and the column-major one used by OptiX buffers,
see tests/OCtx2dTest.cc tests/OCtx3dTest.cc 

**/


RT_PROGRAM void raygen()
{
/*
    int  iphi = int(launch_index.x) ;                    
    int  ithe = int(launch_index.y) ;  
    int  icat = int(launch_index.z) ;  

    rtPrintf("//raygen launch_index (%d %d %d) launch_dim (%d %d %d) \n",
        launch_index.x, launch_index.y, launch_index.z,
        launch_dim.x, launch_dim.y, launch_dim.z );
*/

    float phi = (float(launch_index.x)+0.5f)/float(launch_dim.x) ; 
    float the = (float(launch_index.y)+0.5f)/float(launch_dim.y) ; 
   
    int tex_id = texid_buffer[icat].x ; 
    float val = rtTex2D<float>( tex_id, phi, the ); 

    //if( val > 0.5f)
    //rtPrintf("//raygen icat %d tex_id %d val %f \n", icat, tex_id, val ); 

    output_buffer[launch_index] = val ; 
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


