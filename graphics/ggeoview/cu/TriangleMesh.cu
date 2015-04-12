
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtBuffer<float3> vertexBuffer;     

// these buffers could be combined into an int4
rtBuffer<int3> indexBuffer; 
rtBuffer<unsigned int> nodeBuffer; 

rtDeclareVariable(unsigned int, nodeIndex, attribute node_index,);
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void mesh_intersect(int primIdx)
{
    int3 index = indexBuffer[primIdx];
    nodeIndex = nodeBuffer[primIdx];
 
    float3 p0 = vertexBuffer[index.x];
    float3 p1 = vertexBuffer[index.y];
    float3 p2 = vertexBuffer[index.z];

    float3 n;
    float  t, beta, gamma;
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        if(rtPotentialIntersection( t ))
        {
            geometricNormal = normalize(n);
            rtReportIntersection(0);   
            //
            //
            // hmm this needs to report the material index of the primitive, 
            // if more than one material associated to the mesh
            // but are still using separate GIs with one material for each
            //
            //
            // sidedness determination in mesh_intersect ? Nope
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            //
            // Could encode the sidedness into the material index reported
            // as the geometricNormal is already in register here
            // (my optix "materials" are actually pairs of materials 
            // and sometimes surface properties) 
            //
            // would that work ? need to transform normal from object to world space 
            // could double up materials with pairs flipped ? 
            //
            // NOPE, better to defer sidedness checking and most everything else 
            // to closest_hit as there will often be many mesh intersections 
            // reporting different intersections that optix sifts through to
            // find the closest
            //
            // 
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


