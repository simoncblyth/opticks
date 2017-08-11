#include "OpticksCSG.h"

// shape flag enums from npy-
#include "NPart.h"
#include "NCylinder.h"
#include "NSlab.h"
#include "NZSphere.h"

#include <optix_world.h>

#include "quad.h"
#include "Part.h"
#include "Prim.h"

#include "switches.h"
#define DEBUG 1


#include "boolean_solid.h"
//#include "hemi-pmt.h"

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

rtBuffer<Prim>  primBuffer; 
rtBuffer<uint4>  identityBuffer;   // from GMergedMesh::getAnalyticInstanceIdentityBuffer()

rtBuffer<float4> prismBuffer ;   // TODO: migrate prism to use planBuffer

rtBuffer<rtCallableProgramId<unsigned(double,double,double,double*,unsigned)> > solve_callable ;

// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


#include "bbox.h"
#include "postorder.h"

#include "csg_intersect_primitive.h"
#include "csg_intersect_torus.h"
#include "csg_intersect_part.h"
#include "csg_intersect_boolean.h"

#include "intersect_ztubs.h"
#include "intersect_zsphere.h"
#include "intersect_box.h"
#include "intersect_prism.h"


//#include "transform_test.h"
//#include "solve_callable_test.h"


RT_PROGRAM void bounds (int primIdx, float result[6])
{
    //if(primIdx == 0) transform_test();
    //if(primIdx == 0) solve_callable_test();

    if(primIdx == 0)
    {
        unsigned partBuffer_size = partBuffer.size() ;
        unsigned planBuffer_size = planBuffer.size() ;
        unsigned tranBuffer_size = tranBuffer.size() ;

        rtPrintf("## intersect_analytic.cu:bounds pts:%4d pln:%4d trs:%4d \n", partBuffer_size, planBuffer_size, tranBuffer_size ); 
    }


    optix::Aabb* aabb = (optix::Aabb*)result;
    *aabb = optix::Aabb();

    uint4 identity = identityBuffer[instance_index] ;  // instance_index from OGeo is 0 for non-instanced

    const Prim prim    = primBuffer[primIdx]; 
    unsigned primFlag    = prim.primFlag() ;  

    if(primFlag == CSG_FLAGNODETREE || primFlag == CSG_FLAGINVISIBLE )  
    {
        csg_bounds_prim(primIdx, prim, aabb); 
    }
    else if(primFlag == CSG_FLAGPARTLIST)  
    {
        unsigned partOffset  = prim.partOffset() ;  
        unsigned numParts    = prim.numParts() ; 

        for(unsigned int p=0 ; p < numParts ; p++)
        { 
            Part pt = partBuffer[partOffset + p] ; 
            unsigned typecode = pt.typecode() ; 

            identity.z = pt.boundary() ;  // boundary from partBuffer (see ggeo-/GPmt)
            // ^^^^ needed ? why not other branch ?

            if(typecode == CSG_PRISM) 
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
    const Prim& prim    = primBuffer[primIdx]; 

    unsigned partOffset  = prim.partOffset() ;  
    unsigned numParts    = prim.numParts() ; 
    unsigned primFlag    = prim.primFlag() ;  

    uint4 identity = identityBuffer[instance_index] ; 


    if(primFlag == CSG_FLAGNODETREE)  
    { 
        Part pt0 = partBuffer[partOffset + 0] ; 

        identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary

        evaluative_csg( prim, identity );
        //intersect_csg( prim, identity );

    }
    else if(primFlag == CSG_FLAGINVISIBLE)
    {
        // do nothing : report no intersections for primitives marked with primFlag CSG_FLAGINVISIBLE 
    }  
    else if(primFlag == CSG_FLAGPARTLIST)  
    {
        for(unsigned int p=0 ; p < numParts ; p++)
        {  
            Part pt = partBuffer[partOffset + p] ; 

            identity.z = pt.boundary() ;   

            unsigned typecode = pt.typecode() ; 

            switch(typecode)
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


