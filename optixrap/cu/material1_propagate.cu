/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <optix.h>
#include "PerRayData_propagate.h"
#include "wavelength_lookup.h"

//attributes set by TriangleMesh.cu:mesh_intersect 

rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );

rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
rtDeclareVariable(float,                  t, rtIntersectionDistance, );


/**
Quoting from optix_device.h:rtTransformPoint

    rtTransformPoint transforms a point using the current
    active transformation stack.

    During traversal, intersection and any-hit programs, 
    the current ray will be located in object space.
    During ray generation, closest-hit and miss programs, the current ray
    will be located in world space.  

    For ray generation and miss programs, the transform will 
    always be the identity transform.  For traversal, intersection,
    any-hit and closest-hit programs, the transform will be dependent on
    the set of active transform nodes for the current state.

**/

RT_PROGRAM void closest_hit_propagate()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 
     float cos_theta = dot(n,ray.direction); 

     prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it

     unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ; 
     prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;   
     prd.identity = instanceIdentity ; 
     prd.surface_normal = cos_theta > 0.f ? -n : n ;   

     // for angular efficiency 
     const float3 isect = ray.origin + t*ray.direction ; 
     const float3 local_point = rtTransformPoint( RT_WORLD_TO_OBJECT, isect ); 
     //const float3 local_point = normalize(rtTransformPoint( RT_WORLD_TO_OBJECT, isect )); 

#ifdef WITH_DEBUG_BUFFER
     //prd.debug = isect ; 
     prd.debug = local_point ;
#endif
 
#ifdef WITH_ANGULAR
     const float f_theta = acos( local_point.z )/M_PIf;                        // polar 0->pi ->  0->1
     const float f_phi_ = atan2( local_point.y, local_point.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
     const float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  //  
     prd.f_theta = f_theta ; 
     prd.f_phi = f_phi ; 
#endif


//#define WITH_PRINT_IDENTITY_CH 1
#ifdef WITH_PRINT_IDENTITY_CH
     rtPrintf("// material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (%8d %8d %8d %8d) \n", 
        instanceIdentity.x, 
        instanceIdentity.y, 
        instanceIdentity.z, 
        instanceIdentity.w) ;  

     rtPrintf("// material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (%8d %8d %8d %8d) \n", 
        prd.identity.x, 
        prd.identity.y, 
        prd.identity.z, 
        prd.identity.w) ;  
#endif

}

// prd.boundary 
//    * 1-based index with cos_theta signing, 0 means miss
//    * sign identifies which of inner/outer-material is material1/material2 
//    * by virtue of zero initialization, a miss leaves prd.boundary at zero
//
//  cos_theta > 0.f
//        outward going photons, with p.direction in same hemi as the geometry normal
//
//  cos_theta < 0.f  
//        inward going photons, with p.direction in opposite hemi to geometry normal
//
// surface_normal oriented to point from material2 back into material1 
//

