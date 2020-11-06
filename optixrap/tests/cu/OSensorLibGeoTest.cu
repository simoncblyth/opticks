#include <optix_world.h>

using namespace optix;

#include "OSensorLib.hh" 


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

// geometry
rtDeclareVariable(rtObject, top_object, , );

// view param
rtDeclareVariable(float3,   eye, , );
rtDeclareVariable(float3,   U, , );
rtDeclareVariable(float3,   V, , );
rtDeclareVariable(float3,   W, , );
rtDeclareVariable(float,    scene_epsilon, , );
rtDeclareVariable(unsigned, radiance_ray_type, , );

// communication from geometry intersect to closest hit  
rtDeclareVariable(float3,   shading_normal,     attribute shading_normal, );
rtDeclareVariable(unsigned, intersect_identity, attribute intersect_identity, );

// communication from closest hit to raygen
rtDeclareVariable(PerRayData, prd,    rtPayload, );
rtDeclareVariable(optix::Ray, raycur, rtCurrentRay, );
rtDeclareVariable(float,      t,      rtIntersectionDistance, );

// render results, communication from raygen out to C++
rtBuffer<uchar4, 2>   pixels_buffer;
rtBuffer<float4, 2>   posi_buffer;


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



RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}
RT_PROGRAM void closest_hit()
{
    const float3 isect = raycur.origin + t*raycur.direction ; 

    // shade by the normal to the intersected point  
    //const float3 local_normal = normalize(rtTransformNormal(RT_WORLD_TO_OBJECT, shading_normal)) ;  
    //prd.result = local_normal*0.5f + 0.5f;

    // shade by angular efficiency 
    const float3 local_point = normalize(rtTransformPoint( RT_WORLD_TO_OBJECT, isect ));
    const float f_theta = acos( local_point.z )/M_PIf;                        // polar 0->pi ->  0->1
    const float f_phi_ = atan2( local_point.y, local_point.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
    const float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  //  

    // see okg/SphereOfTransforms.cc
    //unsigned itheta       = ( intersect_identity & 0x000000ff ) >> 0 ;
    //unsigned iphi         = ( intersect_identity & 0x0000ff00 ) >> 8 ;
    unsigned sensor_index   = ( intersect_identity & 0xffff0000 ) >> 16  ;  
    const int category = OSensorLib_category( sensor_index ) ; 
    const float eff    = OSensorLib_angular_efficiency(category, f_phi, f_theta); 

    prd.result = make_float3( eff, eff, eff );
    prd.posi = make_float4( isect, __uint_as_float(intersect_identity) );  
}
RT_PROGRAM void miss()
{   
    prd.result = make_float3(0.9f, 0.9f, 0.9f) ;  // light grey 
    prd.posi = make_float4(0.f,0.f,0.f, __uint_as_float(0u));
}


