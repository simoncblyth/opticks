#include "scuda.h"
#include "squad.h"
#include <optix_device.h>

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         tmin, , );
rtDeclareVariable(unsigned,      radiance_ray_type, , );
rtDeclareVariable(unsigned,      cameratype, , );
rtDeclareVariable(unsigned,      raygenmode, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,      t, rtIntersectionDistance, );


// TODO: replace these "real" optix buffer which shims over CUDA buffers just like the geometry
rtBuffer<uchar4, 2>   pixels_buffer;
rtBuffer<float4, 2>   posi_buffer;
rtBuffer<quad4,  2>   isect_buffer;


static __device__ __inline__ uchar4 make_color(const float3& c)
{
    return make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),   
                        static_cast<unsigned char>(__saturatef(c.z)*255.99f),   
                        255u);                                                 
}


struct PerRayData
{
    float3   result;
    int      mode ; 
    float4   posi ; 
};

rtDeclareVariable(float3, position,         attribute position, );  
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );  
rtDeclareVariable(unsigned,  intersect_identity,   attribute intersect_identity, );  

rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(rtObject,      top_object, , );


RT_PROGRAM void raygen()
{
    PerRayData prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ; 
    prd.posi = make_float4( 0.f, 0.f, 0.f, 0.f ); 
    prd.mode = 0 ; 

    const float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;
    const float3 dxyUV = d.x*U + d.y*V ; 
    //                       cameratype     0u perspective           1u orthographic
    const float3 origin    = cameratype == 0u ? eye                    : eye + dxyUV    ; 
    const float3 direction = cameratype == 0u ? normalize( dxyUV + W ) : normalize( W ) ; 

    optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, tmin, RT_DEFAULT_MAX) ; 
    rtTrace(top_object, ray, prd);

    const bool yflip = true ; 
    uint2 index = make_uint2( launch_index.x , yflip ? launch_dim.y - 1u - launch_index.y : launch_index.y );   
    pixels_buffer[index] = make_color( prd.result ) ; 
    posi_buffer[index] = prd.posi ; 

    quad4 isect ; 
    isect.q0.f.x = prd.result.x ; 
    isect.q0.f.y = prd.result.y ; 
    isect.q0.f.z = prd.result.z ;
    isect.q0.f.w = 0.f ;

    isect.q1.f.x = prd.posi.x ; 
    isect.q1.f.y = prd.posi.y ; 
    isect.q1.f.z = prd.posi.z ;
    isect.q1.f.w = prd.posi.w ;

    isect.q2.f.x = origin.x ; 
    isect.q2.f.y = origin.y ; 
    isect.q2.f.z = origin.z ;
    isect.q2.f.w = tmin ;

    isect.q3.f.x = direction.x ; 
    isect.q3.f.y = direction.y ; 
    isect.q3.f.z = direction.z ;
    isect.q3.i.w = prd.mode ;

    isect_buffer[index] = isect ; 

#ifdef DEBUG_SIX
    //rtPrintf("//DEBUG_SIX/OptiXTest.cu:raygen prd.mode %d \n", prd.mode ); 
#endif

}

RT_PROGRAM void miss()
{
    //prd.result = make_float3(0.5f, 1.f, 1.f) ;  // cyan
    prd.result = make_float3(1.f, 1.f, 1.f) ;
    prd.mode = 1 ; 
}

RT_PROGRAM void closest_hit()
{
    prd.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
    float3 isect = ray.origin + t*ray.direction ;
    prd.posi = make_float4( isect, __uint_as_float(intersect_identity) );
    prd.mode = 2 ; 
}

