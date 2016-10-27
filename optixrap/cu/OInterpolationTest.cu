
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#include "boundary_lookup.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtBuffer<float4,2>  out_buffer;



RT_PROGRAM void OInterpolationTest()
{
    unsigned nj = BOUNDARY_NUM_MATSUR ;
    unsigned nk = BOUNDARY_NUM_FLOAT4 ;

    uint2 out_index ; 
    int w = int(launch_index.x) ;  // 0:39 wavelength sample index
    int i = int(launch_index.y) ;  // 0:123 bnd index

    out_index.x = w ;  
    float nm = boundary_domain.x + w*boundary_domain.z ;


    for(unsigned j=0 ; j < nj ; j++)
    {
        unsigned line = i*nj + j ;  
        for(unsigned k=0 ; k < nk ; k++)
        {
           out_index.y = boundary_lookup_linek(line, k );  
           out_buffer[out_index] = boundary_lookup(nm, line, k ) ; 
        }
    }

    // NB 
    //    out_buffer.x same dim as launch.x (width), 
    //    out_buffer.y eight times larger than launch.y (height) to match the above loops

}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}

/*

       #float4
            |     ___ wavelength samples
            |    /
   (123, 4, 2, 39, 4)
    |    |          \___ float4 props        
  #bnd   | 
         |
    omat/osur/isur/imat  
*/


