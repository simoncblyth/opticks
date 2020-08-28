#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtBuffer<uchar4, 2> out_buffer ; 
rtBuffer<float4, 2> dbg_buffer ; 
rtBuffer<float4, 2> pos_buffer ; 

struct PerRayData
{ 
    float3 result;
    float  t ;
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned,      radiance_ray_type, , );

rtDeclareVariable(rtObject,      top_object, , );

// attribute enables connection between geometry intersect and closest hit 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

rtDeclareVariable(float, intersection_distance, rtIntersectionDistance, ); 
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(int4,   tex_param, , );
rtDeclareVariable(float4, tex_domain, , );



// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  // R
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  // G 
                               static_cast<unsigned char>(__saturatef(c.z)*255.99f),  // B 
                               255u);                                                 // A 
}

// Returns shading normal as the surface shading result
RT_PROGRAM void closest_hit_radiance0()
{
    float3 norm = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)) ; 
    //float3 result = norm*0.5f + 0.5f;

    float f_theta = acos( norm.z )/M_PIf;                 // polar 0->pi ->  0->1
    float f_phi_ = atan2( norm.y, norm.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
    float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  // 

    int texture_id = tex_param.w ; 
    unsigned layer = 0u ; 
    uchar4 val = rtTex2DLayered<uchar4>( texture_id, f_phi, f_theta, layer );
    float3 result = make_float3( float(val.x)/255.99f,  float(val.y)/255.99f,  float(val.z)/255.99f ) ;  

#ifdef DUMP
    rtPrintf("//UseOptiXTextureLayeredOKImgGeo.cu:raygen closest_hit_radiance0 tex_param (%d %d %d %d) norm (%f %f %f) f_theta/phi (%f %f)  \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         norm.x, 
         norm.y, 
         norm.z, 
         f_theta, 
         f_phi,
     );
#endif

    prd.result = result ; 
    prd.t = intersection_distance ; 
}

RT_PROGRAM void miss()
{
    prd.result = make_float3(1.f, 0.f, 1.f) ;
    prd.t = 0.f ; 
}

RT_PROGRAM void raygen_reproduce_texture()
{
    float2 fd = make_float2(launch_index)/make_float2(launch_dim);   // 0->1 
    int texture_id = tex_param.w ; 
    uchar4 val = rtTex2DLayered<uchar4>( texture_id, fd.x, fd.y, 0u );
    float3 result = make_float3( float(val.x)/255.99f,  float(val.y)/255.99f,  float(val.z)/255.99f ) ;  
    float3 pos = make_float3( fd.x, fd.y, 0.f) ; 

    dbg_buffer[launch_index] = make_float4( result, 1.f );  
    out_buffer[launch_index] = make_color( result ) ;
    pos_buffer[launch_index] = make_float4( pos, 0.f );

#ifdef DUMP
    rtPrintf("//raygen_reproduce_texture launch_index.xy (%u %u) d (%f %f)  \n", 
         launch_index.x, 
         launch_index.y, 
         fd.x, 
         fd.y
     );
#endif
}

RT_PROGRAM void raygen()
{
    float2 fd = make_float2(launch_index)/make_float2(launch_dim);   // 0->1 
    float2 d = fd*2.f - 1.f ;   // -1 -> 1

    PerRayData prd;
    prd.result = make_float3( 1.f, 1.f, 1.f ) ;
    prd.t = 0.f ; 

    optix::Ray ray ; 
    float3 ray_origin    = eye + d.x*U + d.y*V ; // orthographic
    float3 ray_direction = normalize(W)        ;   
    ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
    rtTrace(top_object, ray, prd);
    float3 pos = ray_origin + prd.t*ray_direction ; 

    out_buffer[launch_index] = make_color( prd.result );
    dbg_buffer[launch_index] = make_float4( prd.result, 1.f );  
    pos_buffer[launch_index] = make_float4( pos, prd.t );
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}

