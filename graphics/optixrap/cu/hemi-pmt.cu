// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);
rtBuffer<float4> analyticBuffer; 
rtBuffer<uint4>  identityBuffer; 


// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



static __device__
void intersect_tubs(int primIdx)
{
  const float4 param0 = analyticBuffer[4*primIdx+0];  
  const float4 param1 = analyticBuffer[4*primIdx+1];  
  const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced

  float3 center = make_float3(param0);
  float radius = param0.w;
  float sizeZ = param1.x ;  

};


template<bool use_robust_method>
static __device__
void intersect_sphere(int primIdx)
{
  const float4 param0 = analyticBuffer[4*primIdx+0];  
  const float4 param1 = analyticBuffer[4*primIdx+1];  

  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
  const uint4 identity = identityBuffer[primIdx] ;  // just primIdx for non-instanced THIS IS WRONG 

  float3 center = make_float3(param0);
  float radius = param0.w;

  float3 O = ray.origin - center;
  float3 D = ray.direction;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;

  if(disc > 0.0f)
  {
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);

    bool do_refine = false;

    float root11 = 0.0f;

    if(use_robust_method && fabsf(root1) > 10.f * radius) 
    {
        do_refine = true;
    }

    if(do_refine) 
    {
        // refine root1
        float3 O1 = O + root1 * ray.direction;
        b = dot(O1, D);
        c = dot(O1, O1) - radius*radius;
        disc = b*b - c;

        if(disc > 0.0f) 
        {
            sdisc = sqrtf(disc);
            root11 = (-b - sdisc);
        }
    }

    bool check_second = true;
    if( rtPotentialIntersection( root1 + root11 ) ) 
    {
        shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
        instanceIdentity = identity ; 

        if(rtReportIntersection(0))  // material index 0
             check_second = false;
    } 
    if(check_second) 
    {
        float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
        if( rtPotentialIntersection( root2 ) ) 
        {
            shading_normal = geometric_normal = (O + root2*D)/radius;
            instanceIdentity = identity ; 

            rtReportIntersection(0);   // material index 0 
        }
    }
  }
}


RT_PROGRAM void intersect(int primIdx)
{
  const float4 min = analyticBuffer[4*primIdx+2];  
  int typecode = __float_as_int(min.w); 
  switch(typecode)
  {
     case 1:
             intersect_sphere<false>(primIdx);
             break ; 
     case 2:
             intersect_tubs(primIdx);
             break ; 
  }
}



RT_PROGRAM void bounds (int primIdx, float result[6])
{
  const float4 min = analyticBuffer[4*primIdx+2];  // 4 quads of buffer for each part
  const float4 max = analyticBuffer[4*primIdx+3]; 

  result[0] = min.x ;    
  result[1] = min.y ;    
  result[2] = min.z ;

  result[3] = max.x ;    
  result[4] = max.y ;    
  result[5] = max.z ;
}

