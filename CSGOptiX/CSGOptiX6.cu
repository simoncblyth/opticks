/**
CSGOptiX6.cu : incomplete backwards compat for new geometry model
=====================================================================

NOTE THAT MANY OF THE BELOW HEADERS NOT USED AS SIMULATION NOT BROUGHT TO 6 
BUT THE UNUSED HEADERS ARE HELPFUL SO THAT COMPILATION ERRORS FROM 6 
ARE SIMILAR TO THOSE FROM 7 : FOR EARLY WARNING ON LAPTOP WHICH CANNOT RUN 7 

TODO: COMBINE GEOM INTO HERE TO MAKE 6 MORE LIKE 7 


**/

#include "scuda.h"
#include "squad.h"


#include "sqat4.h"
#include "sphoton.h"

#include "qstate.h"
#include "qsim.h"
#include "qevent.h"




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


// NB: cannot replace these "real" optix buffer as only input buffers can shims over CUDA buffers like geometry buffers

rtBuffer<uchar4, 2>   pixel_buffer;   // formerly pixels_buffer
rtBuffer<float4, 2>   isect_buffer;   // formerly posi_buffer
rtBuffer<quad4,  2>   photon_buffer;  // formerly isect_buffer


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
    float4   isect ; 
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
    prd.isect  = make_float4( 0.f, 0.f, 0.f, 0.f ); 
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
    pixel_buffer[index] = make_color( prd.result ) ; 
    isect_buffer[index] = prd.isect ; 

    quad4 photon ; 
    photon.q0.f.x = prd.result.x ; 
    photon.q0.f.y = prd.result.y ; 
    photon.q0.f.z = prd.result.z ;
    photon.q0.f.w = 0.f ;

    photon.q1.f.x = prd.isect.x ; 
    photon.q1.f.y = prd.isect.y ; 
    photon.q1.f.z = prd.isect.z ;
    photon.q1.f.w = prd.isect.w ;

    photon.q2.f.x = origin.x ; 
    photon.q2.f.y = origin.y ; 
    photon.q2.f.z = origin.z ;
    photon.q2.f.w = tmin ;

    photon.q3.f.x = direction.x ; 
    photon.q3.f.y = direction.y ; 
    photon.q3.f.z = direction.z ;
    photon.q3.i.w = prd.mode ;

    photon_buffer[index] = photon ; 

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
    prd.isect = make_float4( isect, __uint_as_float(intersect_identity) );
    prd.mode = 2 ; 
}

