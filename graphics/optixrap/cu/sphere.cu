// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
rtDeclareVariable(unsigned int,  instanceIdx, , );


rtBuffer<uint4> identityBuffer; 

// attribute variables must be set 
// inbetween rtPotentialIntersection and rtReportIntersection
// they provide communication from intersection program to closest hit program
 
rtDeclareVariable(unsigned int, nodeIndex,     attribute node_index,);
rtDeclareVariable(unsigned int, boundaryIndex, attribute boundary_index,);
rtDeclareVariable(unsigned int, sensorIndex,   attribute sensor_index,);


rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );




template<bool use_robust_method>
static __device__
void intersect_sphere(int primIdx)
{
  //uint4 identity = identityBuffer[primIdx];
  uint4 identity = identityBuffer[2];

  float3 center = make_float3(sphere);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere.w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);

    bool do_refine = false;

    float root11 = 0.0f;

    if(use_robust_method && fabsf(root1) > 10.f * radius) {
      do_refine = true;
    }

    if(do_refine) {
      // refine root1
      float3 O1 = O + root1 * ray.direction;
      b = dot(O1, D);
      c = dot(O1, O1) - radius*radius;
      disc = b*b - c;

      if(disc > 0.0f) {
        sdisc = sqrtf(disc);
        root11 = (-b - sdisc);
      }
    }

    bool check_second = true;
    if( rtPotentialIntersection( root1 + root11 ) ) {
      shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;

      nodeIndex = identity.x ;
      boundaryIndex = identity.z ;
      sensorIndex = identity.w ;

      if(rtReportIntersection(0))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
      if( rtPotentialIntersection( root2 ) ) {
        shading_normal = geometric_normal = (O + root2*D)/radius;

        nodeIndex = identity.x ;
        boundaryIndex = identity.z ;
        sensorIndex = identity.w ;

        rtReportIntersection(0);
      }
    }
  }
}


RT_PROGRAM void intersect(int primIdx)
{
  intersect_sphere<false>(primIdx);
}


RT_PROGRAM void robust_intersect(int primIdx)
{
  intersect_sphere<true>(primIdx);
}


RT_PROGRAM void bounds (int primIdx, float result[6])
{
  const float3 cen = make_float3( sphere );
  const float3 rad = make_float3( sphere.w );

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if( rad.x > 0.0f  && !isinf(rad.x) ) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}

