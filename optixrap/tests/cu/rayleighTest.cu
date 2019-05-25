
#include <curand_kernel.h>

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<curandState, 1>       rng_states ;

rtBuffer<float4,2>  rayleigh_buffer;


// HMM : NOT GOOD THAT SO MUCH BAGGAGE NEEDED TO TEST rayleigh.h

#include "cu/quad.h"
#include "cu/boundary_lookup.h"

rtBuffer<uint4>                optical_buffer; 

#include "cu/state.h"
#include "cu/photon.h"
#include "cu/rayleigh.h"


RT_PROGRAM void rayleighTest()
{
    unsigned long long generate_id = launch_index.x ;
    //unsigned long long generate_offset = unsigned(generate_id)*4 ;
    curandState rng = rng_states[generate_id];

    Photon o, p ; 
    o.direction.x = 1.f ; 
    o.direction.y = 0.f ; 
    o.direction.z = 0.f ; 

    o.polarization.x = 0.f ; 
    o.polarization.y = 1.f ; 
    o.polarization.z = 0.f ; 

    p.direction.x = o.direction.x ; 
    p.direction.y = o.direction.y ; 
    p.direction.z = o.direction.z ; 

    p.polarization.x = o.polarization.x ; 
    p.polarization.y = o.polarization.y ; 
    p.polarization.z = o.polarization.z ; 
 
    rayleigh_scatter( p, rng );

    rtPrintf("//rayleighTest generate_id %llu \n", generate_id ); 

    uint2 u0 = make_uint2( unsigned(generate_id), 0u ) ;
    uint2 u1 = make_uint2( unsigned(generate_id), 1u ) ;
    uint2 u2 = make_uint2( unsigned(generate_id), 2u ) ;
    uint2 u3 = make_uint2( unsigned(generate_id), 3u ) ;

    rayleigh_buffer[u0] = make_float4( o.direction.x,     o.direction.y,     o.direction.z,     0.f );
    rayleigh_buffer[u1] = make_float4( o.polarization.x,  o.polarization.y,  o.polarization.z,  0.f );
    rayleigh_buffer[u2] = make_float4( p.direction.x,     p.direction.y,     p.direction.z,     0.f );
    rayleigh_buffer[u3] = make_float4( p.polarization.x,  p.polarization.y,  p.polarization.z,  0.f );
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


