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
void intersect_tubs(const float4& zrg, const float4& q0, const float4& q1, const uint4& identity )
{
  //float3 position = make_float3(q0.x, q0.y, q0.z);
  //float radius = q0.w;
  //float sizeZ =  q1.x ;  
  //
};


template<bool use_robust_method>
static __device__
void intersect_sphere(const float4& zrg, const float4& q0, const uint4& identity)
{
  float3 center = make_float3(q0);
  float radius = q0.w;

  float3 O = ray.origin - center;
  float3 D = ray.direction;

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
        bool do_refine = use_robust_method && fabsf(root1) > 10.f * radius ;  // long ray 

        if(do_refine) // refine root1
        {
            float3 O1 = O + root1 * ray.direction;  //  move origin along to 1st intersection point
            b = dot(O1, D);
            c = dot(O1, O1) - radius*radius;
            disc = b*b - c;
            if(disc > 0.0f) 
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }
        float3 P = ray.origin + (root1 + root11)*ray.direction ;  
        if( P.z > zrg.x && P.z < zrg.y )
        {
            bool check_second = true;
            if( rtPotentialIntersection( root1 + root11 ) ) 
            {
                shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
                instanceIdentity = identity ; 
                if(rtReportIntersection(0)) check_second = false;
            } 
            if(check_second) 
            {
                float root2 = (-b + sdisc) + (do_refine ? root11 : 0.f);   // unconfirmed change root1 -> root11
                P = ray.origin + root2*ray.direction ;  
                if( P.z > zrg.x && P.z < zrg.y )
                { 
                    if( rtPotentialIntersection( root2 ) ) 
                    {
                        shading_normal = geometric_normal = (O + root2*D)/radius;
                        instanceIdentity = identity ; 
                        rtReportIntersection(0);   // material index 0 
                    }
                }
            }
        }
    }
}

static __device__
void intersect_aabb(const float4& q2, const float4& q3, const uint4& identity)
{
  const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
  const float3 max_ = make_float3(q3.x, q3.y, q3.z); 

  float3 t0 = (min_ - ray.origin)/ray.direction;
  float3 t1 = (max_ - ray.origin)/ray.direction;

  // slab method 
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  float3 n = make_float3(0.f);  

  if(tmin <= tmax) 
  {
      if(rtPotentialIntersection(tmin))
      {
          // hmm what about inside box ?
          if(     tmin == near.x) n.x = 1. ;
          else if(tmin == near.y) n.y = 1. ;
          else if(tmin == near.z) n.z = 1. ;

          shading_normal = geometric_normal = n ;
          instanceIdentity = identity ;
          rtReportIntersection(0);   // material index 0 
      } 
  }
}

static __device__
bool intersect_aabb(const float4& q2, const float4& q3)
{
    const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
    const float3 max_ = make_float3(q3.x, q3.y, q3.z); 
    float3 t0 = (min_ - ray.origin)/ray.direction;
    float3 t1 = (max_ - ray.origin)/ray.direction;
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf( near );
    float tmax = fminf( far );
    return tmin <= tmax ;
}



RT_PROGRAM void intersect(int primIdx)
{
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 
  const uint4& identity = identityBuffer[primIdx] ; 
  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced

  //rtPrintf("intersect primIdx %d numParts %u \n", primIdx, numParts );

  for(unsigned int p=0 ; p < numParts ; p++)
  {  
      unsigned int partIdx = solid.x + p ;  

      const float4& q0 = partBuffer[4*partIdx+0];  
      const float4& q1 = partBuffer[4*partIdx+1];  
      const float4& q2 = partBuffer[4*partIdx+2] ;
      const float4& q3 = partBuffer[4*partIdx+3]; 

      float4 zrange = make_float4( q2.z , q3.z, 0.f, 0.f ) ;
      int typecode = __float_as_int(q2.w); 

      switch(typecode)
      {
          case 0:
                intersect_aabb(q2, q3, identity);
                break ; 
          case 1:
                intersect_sphere<true>(zrange, q0, identity);
                break ; 
          case 2:
                intersect_tubs(zrange,q0,q1, identity);
                break ; 
      }
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

