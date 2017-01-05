
#include <optix_world.h>
#include "quad.h"
#include "hemi-pmt.h"
#include "math_constants.h"

using namespace optix;

//#include "wavelength_lookup.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);

// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



#define DEBUG 1


template<bool use_robust_method>
static __device__
void intersect_zsphere(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
{

  float3 center = make_float3(q0.f);
  float radius = q0.f.w;

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
        bool check_second = true;
        if( P.z >= q2.f.z && P.z <= q3.f.z )
        {
            if( rtPotentialIntersection( root1 + root11 ) ) 
            {
                shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
                instanceIdentity = identity ; 
                if(rtReportIntersection(0)) check_second = false;
            } 
        }

        if(check_second) 
        {
            float root2 = (-b + sdisc) + (do_refine ? root11 : 0.f);   // unconfirmed change root1 -> root11
            P = ray.origin + root2*ray.direction ;  
            if( P.z >= q2.f.z && P.z <= q3.f.z )
            { 
                if( rtPotentialIntersection( root2 ) ) 
                {
                    shading_normal = geometric_normal = (O + root2*D)/radius; 
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);   // material index 0 

                    // NB: **NOT** negating normal when inside as that 
                    //     would break rules regards "solidity" of geometric normals
                    //     normal must depend on geometry at intersection point **only**, 
                    //     with no dependence on ray direction
                }
            }
        }
    }
}




RT_PROGRAM void bounds (int primIdx, float result[6])
{
  // could do offline
  // but this is great place to dump things checking GPU side state
  // as only run once


  const uint4& solid    = solidBuffer[primIdx]; 
  uint4 identity = identityBuffer[instance_index] ; 
  unsigned int numParts = solid.y ; 

  optix::Aabb* aabb = (optix::Aabb*)result;
  *aabb = optix::Aabb();
  // expand aabb to include all the bbox of the parts 

  for(unsigned int p=0 ; p < numParts ; p++)
  { 
      unsigned int partIdx = solid.x + p ;  

      quad q0, q1, q2, q3 ; 

      q0.f = partBuffer[4*partIdx+0];  
      q1.f = partBuffer[4*partIdx+1];  
      q2.f = partBuffer[4*partIdx+2] ;
      q3.f = partBuffer[4*partIdx+3]; 
      
      int partType = q2.i.w ; 

      identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
/*
      unsigned int boundary = q1.u.z ; 
      rtPrintf("bounds primIdx %u p %u partIdx %u boundary %u identity (%u,%u,%u,%u) partType %d \n", primIdx, p, partIdx, boundary,  
                  identity.x, 
                  identity.y, 
                  identity.z, 
                  identity.w,
                  partType 
              );  

      rtPrintf("q0 %10.4f %10.4f %10.4f %10.4f q1 %10.4f %10.4f %10.4f %10.4f \n",
                  q0.f.x, 
                  q0.f.y, 
                  q0.f.z, 
                  q0.f.w,
                  q1.f.x, 
                  q1.f.y, 
                  q1.f.z, 
                  q1.f.w);

      rtPrintf("q2 %10.4f %10.4f %10.4f %10.4f q3 %10.4f %10.4f %10.4f %10.4f \n",
                  q2.f.x, 
                  q2.f.y, 
                  q2.f.z, 
                  q2.f.w,
                  q3.f.x, 
                  q3.f.y, 
                  q3.f.z, 
                  q3.f.w);
*/

      if(partType == 4) 
      {
          make_prism(q0.f, aabb) ;
      }
      else
      {
          aabb->include( make_float3(q2.f), make_float3(q3.f) );
      }
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



RT_PROGRAM void intersect(int primIdx)
{
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 

  //const uint4& identity = identityBuffer[primIdx] ; 
  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced

  // try with just one identity per-instance 
  uint4 identity = identityBuffer[instance_index] ; 


  for(unsigned int p=0 ; p < numParts ; p++)
  {  
      unsigned int partIdx = solid.x + p ;  

      quad q0, q1, q2, q3 ; 

      q0.f = partBuffer[4*partIdx+0];  
      q1.f = partBuffer[4*partIdx+1];  
      q2.f = partBuffer[4*partIdx+2] ;
      q3.f = partBuffer[4*partIdx+3]; 

      identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)

      int partType = q2.i.w ; 

      // TODO: use enum
      switch(partType)
      {
          case 0:
                intersect_aabb(q2, q3, identity);
                break ; 
          case 1:
                intersect_zsphere<false>(q0,q1,q2,q3,identity);
                break ; 
          case 2:
                intersect_ztubs(q0,q1,q2,q3,identity);
                break ; 
          case 3:
                intersect_box(q0,q1,q2,q3,identity);
                break ; 
          case 4:
                intersect_prism(q0,q1,q2,q3,identity);
                break ; 

      }
  }

}


