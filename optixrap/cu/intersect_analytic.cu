#include "OpticksCSG.h"
#include "NPart.h"

#include <optix_world.h>

#include "quad.h"
#include "Part.h"

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

rtBuffer<Part> partBuffer; 
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
#include "transform_test.h"

#include "csg_intersect_part.h"
#include "csg_intersect_boolean.h"

#include "intersect_ztubs.h"
#include "intersect_zsphere.h"
#include "intersect_box.h"
#include "intersect_prism.h"


/*
TODO
~~~~~~

* use prim.z for numTran, instead of duplicating primIdx 

*/

RT_PROGRAM void bounds (int primIdx, float result[6])
{
    if(primIdx == 0) 
    { 
        transform_test();
    }

    unsigned tranBuffer_size = tranBuffer.size() ;
    const uint4& prim    = primBuffer[primIdx]; 

    unsigned partOffset  = prim.x ;  
    unsigned numParts    = prim.y ; 
    unsigned primFlag    = prim.w ;  

    unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
    unsigned numNodes = TREE_NODES(height) ;      

    rtPrintf("##bounds primIdx %2d partOffset %2d numParts %2d height %2d numNodes %2d tranBuffer_size %3u \n", primIdx, partOffset, numParts, height, numNodes, tranBuffer_size );

    uint4 identity = identityBuffer[instance_index] ;  // instance_index from OGeo is 0 for non-instanced

    optix::Aabb* aabb = (optix::Aabb*)result;
    *aabb = optix::Aabb();

    if(primFlag == CSG_FLAGNODETREE)  
    {
        unsigned nodeIdx = 1 << height ; 
        while(nodeIdx)
        {
            int depth = TREE_DEPTH(nodeIdx) ;
            int elev = height - depth ; 

            Part pt = partBuffer[partOffset+nodeIdx-1];  // nodeIdx is 1-based

            unsigned partType = pt.partType() ; 
            unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
    
            rtPrintf("## bounds nodeIdx %2u depth %2d elev %2d partType %2u gtransformIdx %2u \n", nodeIdx, depth, elev, partType, gtransformIdx );

            if(gtransformIdx == 0)
            {
                switch(partType)
                {
                    case CSG_SPHERE: csg_bounds_sphere(pt.q0, aabb, NULL  );  break ;
                    case CSG_BOX:    csg_bounds_box(pt.q0, aabb, NULL  );     break ;
                    default:                                                  break ; 
                }
            }
            else
            {
                unsigned trIdx = 2*(gtransformIdx-1)  ; 
                if(trIdx >= tranBuffer_size)
                { 
                    rtPrintf("## bounds ABORT trIdx %3u overflows tranBuffer_size %3u \n", trIdx, tranBuffer_size );
                    return ;  
                }
                optix::Matrix4x4 tr = tranBuffer[trIdx] ; 
                switch(partType)
                {
                    case CSG_SPHERE: csg_bounds_sphere(pt.q0, aabb, &tr  );  break ;
                    case CSG_BOX:    csg_bounds_box(   pt.q0, aabb, &tr  );  break ;
                    default:                                                 break ; 
                }
            }

            nodeIdx = nodeIdx & 1 ? nodeIdx >> 1 : (nodeIdx << elev) + (1 << elev) ;
            // see opticks/dev/csg/postorder.py for explanation of bit-twiddling postorder  
        }
    }
    else if(primFlag == CSG_FLAGPARTLIST)  
    {
        for(unsigned int p=0 ; p < numParts ; p++)
        { 
            Part pt = partBuffer[partOffset + p] ; 
            unsigned partType = pt.q2.u.w ; 

            identity.z = pt.q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)

            if(partType == CSG_PRISM) 
            {
                make_prism(pt.q0.f, aabb) ;
            }
            else
            {
                aabb->include( make_float3(pt.q2.f), make_float3(pt.q3.f) );
            }
        } 
    }
    else
    {
        rtPrintf("## intersect_analytic.cu:bounds ABORT BAD primflag %d \n", primFlag );
        return ; 
    }
    rtPrintf("##intersect_analytic.cu:bounds primIdx %d primFlag %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, primFlag, 
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5]
        );

}


/**

identityBuffer
~~~~~~~~~~~~~~~~

* just placeholder zeros for analytic test geometry 

* setting identity.z adopts boundary index from partBuffer, see npy/NPart.hpp for layout (also GPmt)
  at intersections the uint4 identity is copied into the instanceIdentity attribute,
  hence making it available to material1_propagate.cu:closest_hit_propagate
  where crucially the instanceIdentity.z -> boundaryIndex


**/


RT_PROGRAM void intersect(int primIdx)
{
    const uint4& prim    = primBuffer[primIdx]; 

    unsigned partOffset  = prim.x ;  
    unsigned numParts    = prim.y ; 
    unsigned primFlag    = prim.w ;  

    uint4 identity = identityBuffer[instance_index] ; 


    if(primFlag == CSG_FLAGNODETREE)  
    { 
        Part pt = partBuffer[partOffset] ; 

        identity.z = pt.q1.u.z ;        // replace placeholder zero with test analytic geometry root node boundary

        evaluative_csg( prim, identity );
        //intersect_csg( prim, identity );

    }
    else if(primFlag == CSG_FLAGPARTLIST)  
    {
        for(unsigned int p=0 ; p < numParts ; p++)
        {  
            Part pt = partBuffer[partOffset + p] ; 

            identity.z = pt.q1.u.z ;   

            unsigned partType = pt.q2.u.w ; 

            switch(partType)
            {
                case CSG_ZERO:
                    intersect_aabb(pt.q2, pt.q3, identity);
                    break ; 
                case CSG_SPHERE:
                    intersect_zsphere<false>(pt.q0,pt.q1,pt.q2,pt.q3,identity);
                    break ; 
                case CSG_TUBS:
                    intersect_ztubs(pt.q0,pt.q1,pt.q2,pt.q3,identity);
                    break ; 
                case CSG_BOX:
                    intersect_box(pt.q0,identity);
                    break ; 
                case CSG_PRISM:
                    // q0.f param used in *bounds* to construct prismBuffer, which is used within intersect_prism
                    intersect_prism(identity);
                    break ; 
            }
        }
    } 
}


