#include "OpticksCSG.h"
#include "NPart.h"

#include <optix_world.h>

#include "quad.h"
#include "switches.h"
#define DEBUG 1

#include "boolean_solid.h"
#include "hemi-pmt.h"

// CUDART_ defines
#include "math_constants.h"

using namespace optix;

// generated from /Users/blyth/opticks/optixrap/cu by boolean_h.py on Sat Mar  4 20:37:03 2017 
rtDeclareVariable(uint4, packed_boolean_lut_ACloser, , ) = { 0x22121141, 0x00014014, 0x00141141, 0x00000000 } ; 
rtDeclareVariable(uint4, packed_boolean_lut_BCloser, , ) = { 0x22115122, 0x00022055, 0x00133155, 0x00000000 } ; 

static __device__
int boolean_ctrl_packed_lookup( OpticksCSG_t operation, IntersectionState_t stateA, IntersectionState_t stateB, bool ACloser )
{
    const uint4& lut = ACloser ? packed_boolean_lut_ACloser : packed_boolean_lut_BCloser ;
    unsigned offset = 3*(unsigned)stateA + (unsigned)stateB ;   
    unsigned index = (unsigned)operation - (unsigned)CSG_UNION ; 
    return offset < 8 ? (( getByIndex(lut, index) >> (offset*4)) & 0xf) : CTRL_RETURN_MISS ;
}


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float, t_parameter, rtIntersectionDistance, );
rtDeclareVariable(float, propagate_epsilon, , );


rtDeclareVariable(unsigned int, instance_index,  ,);
// optix::GeometryInstance instance_index into the identity buffer, 
// set by oxrap/OGeo.cc, 0 for non-instanced 

rtDeclareVariable(unsigned int, primitive_count, ,);
// TODO: instanced analytic identity, using the above and below solid level identity buffer

rtBuffer<float4> partBuffer; 

rtBuffer<Matrix4x4> tranBuffer; 


rtBuffer<uint4>  primBuffer; 
rtBuffer<uint4>  identityBuffer;   // from GMergedMesh::getAnalyticInstanceIdentityBuffer()
rtBuffer<float4> prismBuffer ;


// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



#include "bbox.h"

#include "csg_intersect_part.h"
#include "csg_intersect_boolean.h"

#include "intersect_ztubs.h"
#include "intersect_zsphere.h"
#include "intersect_box.h"
#include "intersect_prism.h"



RT_PROGRAM void bounds (int primIdx, float result[6])
{
  if(primIdx == 0) 
  { 
      test_tranBuffer();
      test_transform_bbox();
  }

  const uint4& prim    = primBuffer[primIdx]; 
  unsigned partOffset  = prim.x ;  
  unsigned numParts    = prim.y ; 
  // can prim.z be used for numTransforms ? 
  unsigned primFlags   = prim.w ;  

  uint4 identity = identityBuffer[instance_index] ;  // 0 for non-instanced, from OGeo

  optix::Aabb* aabb = (optix::Aabb*)result;
  *aabb = optix::Aabb();

  // expand aabb to include all the bbox of the parts 

  bool is_csg = primFlags == CSG_UNION || primFlags == CSG_INTERSECTION || primFlags == CSG_DIFFERENCE ;  

  if(is_csg)  
  {
      // +0 : csg bbox based on first csg node, the root of tree
      // TODO: need to traverse the tree and include bboxen accounting for node transforms

      quad q2, q3 ; 
      q2.f = partBuffer[4*(partOffset+0)+2];  
      q3.f = partBuffer[4*(partOffset+0)+3];  

      aabb->include( make_float3(q2.f), make_float3(q3.f) );
  }
  else
  {
      for(unsigned int p=0 ; p < numParts ; p++)
      { 
          quad q0, q1, q2, q3 ; 

          q0.f = partBuffer[4*(partOffset+p)+0];  
          q1.f = partBuffer[4*(partOffset+p)+1];  
          q2.f = partBuffer[4*(partOffset+p)+2] ;
          q3.f = partBuffer[4*(partOffset+p)+3]; 
          
          unsigned partType = q2.u.w ; 

          identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)

          if(partType == CSG_PRISM) 
          {
              make_prism(q0.f, aabb) ;
          }
          else
          {
              aabb->include( make_float3(q2.f), make_float3(q3.f) );
          }
      } 
   }

  rtPrintf("##hemi-pmt.cu:bounds primIdx %d is_csg:%d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, is_csg, 
       result[0],
       result[1],
       result[2],
       result[3],
       result[4],
       result[5]
     );

}




RT_PROGRAM void intersect(int primIdx)
{
  const uint4& prim    = primBuffer[primIdx]; 

  unsigned partOffset  = prim.x ;  
  unsigned numParts    = prim.y ; 
  //unsigned primIdx_    = prim.z ;    // <--- looks like its spare, can be used for numTran ?
  unsigned primFlags   = prim.w ;  

  //if(primIdx > 0)
  //rtPrintf("intersect primIdx:%d partOffset(x):%u numParts(y):%u primFlags(w):%u \n", primIdx, partOffset, numParts, primFlags ); 

  uint4 identity = identityBuffer[instance_index] ; 
  // for analytic test geometry (PMT too?) the identityBuffer  
  // is composed of placeholder zeros

  bool is_csg = primFlags == CSG_UNION || primFlags == CSG_INTERSECTION || primFlags == CSG_DIFFERENCE ;  

  // TODO: currently primFlags from prim buffer is just copied from part-typecode of the root node in part buffer, 
  //       so for a partlist that would be one of the primitives, CSG_SPHERE, CSG_BOX etc..
  //       whereas it really refers to the composite, so could use different set of enums perhaps one of:
  //
  //             CSG_PRIMFLAG_TREE 
  //             CSG_PRIMFLAG_PARTLIST   
  //

  if(is_csg)
  { 
      //if(primIdx>0)
      //rtPrintf("intersect(csg) primIdx:%d partOffset(x):%u numParts(y):%u primIdx_(z):%u primFlags(w):%u \n", primIdx, partOffset, numParts, primIdx_, primFlags ); 

      quad q1 ; 
      q1.f = partBuffer[4*(partOffset+0)+1];  
      identity.z = q1.u.z ;        // replace placeholder zero with test analytic geometry boundary

      //intersect_boolean_triplet( prim, identity );
      //intersect_csg( prim, identity );
      evaluative_csg( prim, identity );
      //recursive_csg( prim, identity );
      //intersect_boolean_only_first( prim, identity );
  }
  else
  {
      //if(primIdx>0)
      //rtPrintf("intersect (partlist) primIdx:%d partOffset(x):%u numParts(y):%u primFlags(w):%u \n", primIdx, partOffset, numParts, primFlags ); 
      // partitioned intersect over single basis-shape parts for each "prim"
      for(unsigned int p=0 ; p < numParts ; p++)
      {  
          unsigned int partIdx = partOffset + p ;  

          //rtPrintf("intersect partloop, numParts:%u p:%u partIdx:%u partOffset:%u \n", numParts, p, partIdx, partOffset );

          quad q0, q1, q2, q3 ; 

          q0.f = partBuffer[4*partIdx+0];  
          q1.f = partBuffer[4*partIdx+1];  
          q2.f = partBuffer[4*partIdx+2] ;
          q3.f = partBuffer[4*partIdx+3]; 

          identity.z = q1.u.z ;   

          // identity.z adopts boundary index from partBuffer, see npy/NPart.hpp for layout (also GPmt)
          // at intersections the uint4 identity is copied into the instanceIdentity attribute,
          // hence making it available to material1_propagate.cu:closest_hit_propagate
          // where crucially the instanceIdentity.z -> boundaryIndex

          unsigned partType = q2.u.w ; 

          switch(partType)
          {
              case CSG_ZERO:
                    intersect_aabb(q2, q3, identity);
                    break ; 
              case CSG_SPHERE:
                    intersect_zsphere<false>(q0,q1,q2,q3,identity);
                    break ; 
              case CSG_TUBS:
                    intersect_ztubs(q0,q1,q2,q3,identity);
                    break ; 
              case CSG_BOX:
                    intersect_box(q0,identity);
                    break ; 
              case CSG_PRISM:
                    intersect_prism(q0,q1,q2,q3,identity);
                    break ; 

          }
      }
   } 
}



