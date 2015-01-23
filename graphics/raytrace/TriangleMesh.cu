//usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/geometry_instance/TriangleMesh.cu
/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.

*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtBuffer<float3> vertexBuffer;     
rtBuffer<float3> normalBuffer;
rtBuffer<float3> tangentBuffer;
rtBuffer<float3> bitangentBuffer;
rtBuffer<float2> texCoordBuffer;
rtBuffer<int3> indexBuffer; 
rtDeclareVariable(unsigned int, hasTangentsAndBitangents, ,);

rtDeclareVariable(float2, textureCoordinate, attribute textureCoordinate, ); 
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(float3, tangent, attribute tangent, ); 
rtDeclareVariable(float3, bitangent, attribute bitangent, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void mesh_intersect(int primIdx)
{
    int3 index = indexBuffer[primIdx];

    float3 p0 = vertexBuffer[index.x];
    float3 p1 = vertexBuffer[index.y];
    float3 p2 = vertexBuffer[index.z];

    // Intersect ray with triangle
    float3 n;
    float  t, beta, gamma;
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        if(rtPotentialIntersection( t ))
        {
            if ( normalBuffer.size() == 0 )
            {
                shadingNormal = normalize( n );
            }
            else
            {
                float3 n0 = normalBuffer[index.x];
                float3 n1 = normalBuffer[index.y];
                float3 n2 = normalBuffer[index.z];
                shadingNormal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );

                if(hasTangentsAndBitangents)
                {
                    float3 t0 = tangentBuffer[index.x];
                    float3 t1 = tangentBuffer[index.y];
                    float3 t2 = tangentBuffer[index.z];
                    tangent = normalize( t1*beta + t2*gamma + t0*(1.0f-beta-gamma) );

                    float3 b0 = bitangentBuffer[index.x];
                    float3 b1 = bitangentBuffer[index.y];
                    float3 b2 = bitangentBuffer[index.z];
                    bitangent = normalize( b1*beta + b2*gamma + b0*(1.0f-beta-gamma) );
                }
            }

            geometricNormal = normalize(n);

            // Texture UV coordinates

            if (texCoordBuffer.size() == 0)
            {
                textureCoordinate = make_float2( 0.0f );
            }
            else
            {
                float2 t0 = texCoordBuffer[index.x];
                float2 t1 = texCoordBuffer[index.y];
                float2 t2 = texCoordBuffer[index.z];
                textureCoordinate = t1*beta + t2*gamma + t0*(1.0f-beta-gamma);
            }

            rtReportIntersection(0);
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
