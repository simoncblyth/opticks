#include <optix_world.h>

using namespace optix;

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  // R 
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  // G 
                               static_cast<unsigned char>(__saturatef(c.z)*255.99f),  // B 
                               255u);                                                 // A 
}

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

struct PerRayData
{   
    float3 result;
    float4 posi ;
};


rtDeclareVariable(float3,   eye, , );
rtDeclareVariable(float3,   U, , );
rtDeclareVariable(float3,   V, , );
rtDeclareVariable(float3,   W, , );
rtDeclareVariable(float,    scene_epsilon, , );
rtDeclareVariable(unsigned, radiance_ray_type, , );
rtDeclareVariable(rtObject, top_object, , );

rtBuffer<uchar4, 2>   pixels_buffer;
rtBuffer<float4, 2>   posi_buffer;

// from geometry intersect 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable(unsigned, intersect_identity,   attribute intersect_identity, );

rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(optix::Ray,           raycur, rtCurrentRay, );
rtDeclareVariable(float,                  t, rtIntersectionDistance, );

rtDeclareVariable(int,   texture_id, , );




//#include "OSensorLib.hh" 

RT_PROGRAM void raygen()
{   
    PerRayData prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ;
    prd.posi = make_float4( 0.f, 0.f, 0.f, 0.f ) ;
    
    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;   // -1:1
    
    optix::Ray ray = optix::make_Ray( eye, normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
    rtTrace(top_object, ray, prd);
    
    pixels_buffer[launch_index] = make_color( prd.result ) ;
    posi_buffer[launch_index] = prd.posi ;
}

//float angular_efficiency = OSensorLib_angular_efficiency( category, phi_fraction , theta_fraction ); 


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


RT_PROGRAM void closest_hit()
{
    float3 isect = raycur.origin + t*raycur.direction ; 
    prd.result = normalize(rtTransformNormal(RT_WORLD_TO_OBJECT, shading_normal))*0.5f + 0.5f;
    prd.posi = make_float4( isect, __uint_as_float(intersect_identity) );  
}
RT_PROGRAM void miss()
{   
    prd.result = make_float3(1.f, 1.f, 1.f) ;
    prd.posi = make_float4(0.f,0.f,0.f, __uint_as_float(0u));
}




