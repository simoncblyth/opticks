
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtBuffer<float3> vertexBuffer;     

rtDeclareVariable(unsigned int, instanceIdx,  attribute instance_index,);
rtDeclareVariable(unsigned int, primitiveCount,  attribute primitive_count,);
rtBuffer<uint4> identityBuffer; 


// these buffers could be combined into an int4
rtBuffer<int3> indexBuffer; 

// GNode, GBoundary, GSensor indices  : TODO consolidate into uint4 
rtBuffer<unsigned int> nodeBuffer; 
rtBuffer<unsigned int> boundaryBuffer; 
rtBuffer<unsigned int> sensorBuffer; 


// attribute variables must be set 
// inbetween rtPotentialIntersection and rtReportIntersection
// they provide communication from intersection program to closest hit program
 
rtDeclareVariable(unsigned int, nodeIndex,     attribute node_index,);
rtDeclareVariable(unsigned int, boundaryIndex, attribute boundary_index,);
rtDeclareVariable(unsigned int, sensorIndex,   attribute sensor_index,);
rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);

rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
//rtDeclareVariable(float3, intersectionPosition, attribute intersection_position, ); 


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void mesh_intersect(int primIdx)
{
    int3 index = indexBuffer[primIdx];
 
    //  tried flipping vertex order in unsuccessful attempt to 
    //  get normal shader colors to match OpenGL
    //  observe with touch mode that n.z often small
    //  ... this is just because surfaces are very often vertical
    //
    float3 p0 = vertexBuffer[index.x];
    float3 p1 = vertexBuffer[index.y];  
    float3 p2 = vertexBuffer[index.z];

    float3 n;
    float  t, beta, gamma;
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        if(rtPotentialIntersection( t ))
        {
            // attributes should be set between rtPotential and rtReport
            geometricNormal = normalize(n);
            //instanceIdentity = identityBuffer[instanceIdx*primitiveCount+primIdx] ;  // index just primIdx for non-instanced
            instanceIdentity = identityBuffer[primIdx] ;  // index just primIdx for non-instanced

            nodeIndex = nodeBuffer[primIdx];
            boundaryIndex = boundaryBuffer[primIdx];
            sensorIndex = sensorBuffer[primIdx];
            
            // doing calculation here might (depending on OptiX compiler cleverness) 
            // repeat for all intersections encountered unnecessarily   
            // instead move the calculation into closest_hit
            //
            // intersectionPosition = ray.origin + t*ray.direction  ; 
            // http://en.wikipedia.org/wiki/Line–plane_intersection
          
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


