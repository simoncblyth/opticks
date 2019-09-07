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


RT_PROGRAM void closest_hit_propagate()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 
     float cos_theta = dot(n,ray.direction); 

     prd.cos_theta = cos_theta ;
     prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this
     unsigned int boundaryIndex = instanceIdentity.z ; 
     prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;   
     prd.identity = instanceIdentity ; 
     prd.surface_normal = cos_theta > 0.f ? -n : n ;   
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

