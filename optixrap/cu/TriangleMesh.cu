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
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

// TODO: compare performance as modify int3,float3 -> int4,float4 

// inputs from OGeo
rtBuffer<int3>   indexBuffer; 
rtBuffer<float3> vertexBuffer;     
rtBuffer<uint4>  identityBuffer; 
rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);

// attribute variables communicating from intersection program to closest hit program
// (must be set between rtPotentialIntersection and rtReportIntersection)
rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );



RT_PROGRAM void mesh_intersect(int primIdx)
{
    int3 index = indexBuffer[primIdx];

    float3 p0 = vertexBuffer[index.x];
    float3 p1 = vertexBuffer[index.y];  
    float3 p2 = vertexBuffer[index.z];

    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced

    float3 n;
    float  t, beta, gamma;
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        if(rtPotentialIntersection( t ))
        {
            geometricNormal = normalize(n);
            instanceIdentity = identity ; 

/*
            rtPrintf("mesh_intersect pi:%7d pc:%7u ii:%4u | n:%4u m:%3u b:%3u s:%5u   \n", primIdx, primitive_count, instance_index,
                     instanceIdentity.x,
                     instanceIdentity.y,
                     instanceIdentity.z,
                     instanceIdentity.w
                  );
*/
         
            rtReportIntersection(0);    // material index 0 
        }
    }
}

RT_PROGRAM void mesh_bounds (int primIdx, float result[6])
{  
    const int3 index = indexBuffer[primIdx];

    const float3 v0   = vertexBuffer[ index.x ];
    const float3 v1   = vertexBuffer[ index.y ];
    const float3 v2   = vertexBuffer[ index.z ];

    const float  area = length(cross(v1-v0, v2-v0));

    optix::Aabb* aabb = (optix::Aabb*)result;

    if(area > 0.0f && !isinf(area))
    {
        aabb->m_min = fminf( fminf( v0, v1), v2 );
        aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
    }
    else 
    {
        aabb->invalidate();
    }
}


