// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);

rtBuffer<float4> partBuffer; 
rtBuffer<uint4>  solidBuffer; 
rtBuffer<uint4>  identityBuffer; 


// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



static __device__
bool intersect_tubs(optix::Ray& r, const float4& q0, const float4& q1, float3& n )
{
  //float3 position = make_float3(q0.x, q0.y, q0.z);
  //float radius = q0.w;
  //float sizeZ =  q1.x ;  

  return false ;  
};


template<bool use_robust_method>
static __device__
bool intersect_sphere(optix::Ray&r, const float4& q0, float3& n)
{
  float3 center = make_float3(q0);
  float radius = q0.w;

  float3 O = r.origin - center;
  float3 D = r.direction;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;

 /*
  rtPrintf("intersect_sphere %10.4f %10.4f %10.4f : %10.4f disc %10.4f \n", 
       center.x,  
       center.y,  
       center.z,  
       radius,
       disc);  
  */
  

  if(disc > 0.0f)
  {
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);
    float root11 = 0.0f;
    bool do_refine = false;

    if(use_robust_method && fabsf(root1) > 10.f * radius) 
    {
        do_refine = true;
    }

    if(do_refine) // refine root1
    {
        float3 O1 = O + root1 * r.direction;
        b = dot(O1, D);
        c = dot(O1, O1) - radius*radius;
        disc = b*b - c;

        if(disc > 0.0f) 
        {
            sdisc = sqrtf(disc);
            root11 = (-b - sdisc);
        }
    }

    float t0 = root1 + root11 ;
    float t1 = (-b + sdisc) + (do_refine ? t0 : 0);
    float tmin = fminf( t0, t1 );

    if( tmin < r.tmin  )
    {
        r.tmin = tmin ; 
        n =  (O + tmin*D)/radius;
        return true ; 
    }
  }
  return false ;
}

static __device__
bool intersect_aabb(optix::Ray &r, const float4& q2, const float4& q3, float3& n)
{
  const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
  const float3 max_ = make_float3(q3.x, q3.y, q3.z); 

  float3 t0 = (min_ - r.origin)/r.direction;
  float3 t1 = (max_ - r.origin)/r.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax && tmin <= r.tmax) 
  {
      r.tmin = max(r.tmin,tmin);

      n.x = 0. ; 
      n.y = 0. ; 
      n.z = 0. ; 
      if(     tmin == near.x) n.x = 1. ;
      else if(tmin == near.y) n.y = 1. ;
      else if(tmin == near.z) n.z = 1. ;

      return true;
  }
  return false;
}

static __device__
bool intersect_aabb(optix::Ray &r, const float4& q2, const float4& q3)
{
    const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
    const float3 max_ = make_float3(q3.x, q3.y, q3.z); 
    float3 t0 = (min_ - r.origin)/r.direction;
    float3 t1 = (max_ - r.origin)/r.direction;
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf( near );
    float tmax = fminf( far );
    return tmin <= tmax && tmin <= r.tmax ;
}



RT_PROGRAM void intersect(int primIdx)
{
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 
  const uint4& identity = identityBuffer[primIdx] ; 
  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced

  //rtPrintf("intersect primIdx %d numParts %u \n", primIdx, numParts );

  optix::Ray tmp_ray = ray;
  float3 n ;  

  for(unsigned int p=0 ; p < numParts ; p++)
  {  
      unsigned int partIdx = solid.x + p ;  

      const float4& q0 = partBuffer[4*partIdx+0];  
      const float4& q1 = partBuffer[4*partIdx+1];  
      const float4& q2 = partBuffer[4*partIdx+2] ;
      const float4& q3 = partBuffer[4*partIdx+3]; 

      int typecode = __float_as_int(q2.w); 
      //rtPrintf("intersect p %u partIdx %u type %d \n", p, partIdx, typecode);

      if(!intersect_aabb(tmp_ray, q2, q3)) continue ; 

      switch(typecode)
      {
          case 0:
             intersect_aabb(tmp_ray, q2, q3, n);
             break ; 
          case 1:
             intersect_sphere<true>(tmp_ray, q0, n);
             break ; 
          case 2:
             intersect_tubs(tmp_ray,q0,q1,n);
             break ; 
      }
  }

  float t = tmp_ray.tmin ; 
  if(rtPotentialIntersection(t))
  {
      shading_normal = geometric_normal = n ;
      instanceIdentity = identity ;
      rtReportIntersection(0);   // material index 0 
  } 

}





RT_PROGRAM void bounds (int primIdx, float result[6])
{
  // could do offline
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 

  optix::Aabb* aabb = (optix::Aabb*)result;
  *aabb = optix::Aabb();

  for(unsigned int p=0 ; p < numParts ; p++)
  { 
      unsigned int partIdx = solid.x + p ;  
      const float4& q2 = partBuffer[4*partIdx+2] ;
      const float4& q3 = partBuffer[4*partIdx+3]; 

      aabb->include( make_float3(q2), make_float3(q3) );
  } 

/*
  rtPrintf("bounds primIdx %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, 
       result[0],
       result[1],
       result[2],
       result[3],
       result[4],
       result[5]
     );
*/

}

