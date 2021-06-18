#include <curand_kernel.h>
#include <optix_world.h>

using namespace optix;

//  rng_states rng_skipahead
#include "ORng.hh"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float>  out_buffer;

RT_PROGRAM void rngTest()
{
    unsigned long long photon_id = launch_index.x ;
    curandState rng = rng_states[photon_id];
    float u = curand_uniform(&rng);  
    out_buffer[photon_id] = u ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}

