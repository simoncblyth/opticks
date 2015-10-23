#include <optix.h>
#include <optix_math.h>

#include "PerRayData_radiance.h"
//#include "wavelength_lookup.h"

//geometric_normal is set by the closest hit intersection program 
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );

rtDeclareVariable(float3, contrast_color, , );

rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int,  touch_mode, , );


RT_PROGRAM void closest_hit_radiance()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

  // const float3 n = normalize(rtTransformNormal(RT_WORLD_TO_OBJECT, geometricNormal)) ; 
  // const float3 n = normalize(geometricNormal) ; 

  // currently lambertian from all the above three looks the same
  // because currently have no transforms in play and are 
  // doing everything in world frame  

  const float cos_theta = dot(n,ray.direction);


 /*
  if(touch_mode)
  {
      // n.z often coming out very small, ~1e-9 
      //     this is just due to there being lots of vertical surfaces
      //     so the surface normal has no up-down component
      // 
      //  click on a PMT, and the many triangles oriented in all directions will give appropriate normals
      //
      rtPrintf("(touch)material1_radiance.cu geometricNormal  %10.4f %10.4f %10.4f   n %10.4f %10.4f %10.4f  ct %10.4f  \n",
          geometricNormal.x, 
          geometricNormal.y, 
          geometricNormal.z, 
          n.x, 
          n.y, 
          n.z,
          cos_theta  );
      //wavelength_check();
  }
  */


  // normal shader colors dont match what getting with OpenGL normal shader ???
  //  BGRA format in the mix but swapping x and z doesnt cause a match
  //  CCW triangle winding maybe
  //
  // prd.result = make_float3(-n.z*0.5f + 0.5f,-n.y*0.5f + 0.5f, -n.x*0.5f + 0.5f );                         // normal shader
  // prd.result = make_float3(n.x*0.5f + 0.5f, n.y*0.5f + 0.5f, n.z*0.5f + 0.5f );                         // normal shader
  //
  //prd.result = make_float3( 1.f, 0.f, 0.f ); //red
  //prd.result = make_float3( 0.f, 1.f, 0.f ); //green
  //prd.result = make_float3( 0.f, 0.f, 1.f ); //blue
  //prd.result = make_float3(0.5f);            


    prd.result = make_float3( 0.5f*(1.0f-cos_theta) );  // lambertian shader
    prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code

  //prd.result = make_float3( instanceIdentity.x/13000.f ) ;  // nodeIndex 
  //prd.result = make_float3( instanceIdentity.y/250.f ) ;    // meshIndex
  //prd.result = make_float3( instanceIdentity.z/50.f ) ;     // boundaryIndex
  //prd.result = make_float3( instanceIdentity.w/1000.f ) ;   // sensorIndex  : need to use near clipping to see inside the PMTs to see anything

  //prd.result = contrast_color ;   // according to boundary index, currently only one color as only one material ?
  // prd.result = make_float3( boundaryIndex/50.f );  // grey scale according to boundary "boundary" index
  //prd.result = make_float3(0.f);


  // if(cos_theta > 0.0f ) prd.result.x = 0.5f ; 
  //
  //
  //
  // make back faces reddish : given that the "light" is effectively coming from
  // the viewpoint this can probably only happen due to a geometry bug 
  // Nope, no bug needed : just shooting rays from inside objects should do this.
  //
  // * maybe surfaces too close to each other resulting in numerical precision flipping
  //   between alternate closest hit surfaces  
  // * flipped triangle winding order is not impossible
  // 
  // Little red is seen:
  //
  // * small red triangles at ends of some struts/ribs on top of AD
  // * when enter inside a PMT, see a concentric circle bullseye red/white pattern
  //   no problem is apparent for the external view of the PMT 
  // * from inside calibration assemblies quite a lot of speckly red/black
  // 



}


