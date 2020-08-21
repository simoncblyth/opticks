#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtBuffer<uchar4, 2> out_buffer ; 
rtBuffer<float4, 2> dbg_buffer ; 
rtBuffer<float4, 2> pos_buffer ; 


struct PerRayData_radiance
{ 
    float3 result;
    float  intersection_distance ;
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
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

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
    //float3 result = norm ; 

    float f_theta = acos( norm.z )/M_PIf;                 // polar 0->pi ->  0->1
    float f_phi_ = atan2( norm.y, norm.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
    float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ; 

    float tx = float(tex_param.y)*f_theta ;   // tex_param ( 1 4096 8192 1  ) 
    float ty = float(tex_param.z)*f_phi ; 

    int texture_id = tex_param.w ; 
    uchar4 val = rtTex2DLayered<uchar4>( texture_id, tx, ty, 0u );
    float3 result = make_float3( float(val.x)/255.99f,  float(val.y)/255.99f,  float(val.z)/255.99f ) ;  

/*
    rtPrintf("//UseOptiXTextureLayeredOKImgGeo.cu:raygen closest_hit_radiance0 tex_param (%d %d %d %d) norm (%f %f %f) f_theta/phi (%f %f) txty (%f %f)  \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         norm.x, 
         norm.y, 
         norm.z, 
         f_theta, 
         f_phi,
         tx,
         ty
         //val.x, 
         //val.y, 
         //val.z 
     );
*/

    prd_radiance.result = result ; 
    prd_radiance.intersection_distance = intersection_distance ; 
}

RT_PROGRAM void miss()
{
    prd_radiance.result = make_float3(1.f, 0.f, 1.f) ;
    prd_radiance.intersection_distance = 0.f ; 
}


RT_PROGRAM void raygen()
{
    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

    PerRayData_radiance prd;
    prd.result = make_float3( 0.f, 0.f, 0.f ) ;
    prd.intersection_distance = 0.f ; 

/*
    rtPrintf("//UseOptiXTextureLayeredOKImgGeo.cu:raygen launch_index.xy (%u %u) d (%f %f)  \n", 
         launch_index.x, 
         launch_index.y, 
         d.x, 
         d.y
     );
*/
    optix::Ray ray ; 
    float3 ray_origin    = eye + d.x*U + d.y*V ; // orthographic
    float3 ray_direction = normalize(W)        ;   
    ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
    rtTrace(top_object, ray, prd);
    float3 pos = ray_origin + prd.intersection_distance*ray_direction ; 

    optix::uchar4 val = make_color( prd.result ) ; 
/*
    rtPrintf("//UseOptiXTextureLayeredOKImgGeo.cu:raygen launch_index.xy (%u %u) d (%f %f) val ( %d %d %d %d ) \n", 
         launch_index.x, 
         launch_index.y, 
         d.x, 
         d.y,
         val.x, 
         val.y, 
         val.z, 
         val.w 
       );
*/
    dbg_buffer[launch_index] = make_float4( prd.result, 1.f );  
    out_buffer[launch_index] = val  ;
    pos_buffer[launch_index] = make_float4( pos, prd.intersection_distance );
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}





