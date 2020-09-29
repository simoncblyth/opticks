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

#include "math_constants.h"   // CUDART_ defines

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
rtDeclareVariable(unsigned int, repeat_index, ,);

rtBuffer<Part> partBuffer; 
rtBuffer<Matrix4x4> tranBuffer; 

rtBuffer<Prim>  primBuffer; 

/**
identityBuffer sources depend on geocode of the GMergedMesh
-------------------------------------------------------------

OGeo::makeGeometryTriangles
     GBuffer* rib = mm->getAppropriateRepeatedIdentityBuffer() ;

OGeo::makeTriangulatedGeometry
     GBuffer* id = mm->getAppropriateRepeatedIdentityBuffer();

OGeo::makeAnalyticGeometry
     NPY<unsigned>*  idBuf = mm->getInstancedIdentityBuffer();

**/
rtBuffer<uint4>  identityBuffer;   

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
#include "csg_intersect_part.h"
#include "csg_intersect_boolean.h"


//#define WITH_PRINT 1
//#define WITH_SOLVE 1 
//#define WITH_PARTLIST 1 
//#define WITH_CUBIC 1 
//#define WITH_TORUS 1 
//#define TRANSFORM_TEST 1
//#define CALLABLE_TEST 1

// some resource issue, canna handle both cubic + torus together 
//  currently have only 2 geometry types "analytic" and "triangulated"
//  maybe split off more ??

#ifdef WITH_SOLVE
typedef double Solve_t ;
#include "Solve.h"
#endif

#ifdef WITH_TORUS
typedef double Torus_t ;
#include "csg_intersect_torus.h"
#endif

#ifdef WITH_CUBIC
typedef double Cubic_t ;
#include "csg_intersect_cubic.h"
#endif

#ifdef WITH_PARTLIST
#include "intersect_ztubs.h"
#include "intersect_zsphere.h"
#include "intersect_box.h"
#include "intersect_prism.h"
#endif

#ifdef TRANSFORM_TEST
#include "transform_test.h"
#endif

#ifdef CALLABLE_TEST
#include "solve_callable_test.h"
#endif



RT_PROGRAM void bounds (int primIdx, float result[6])
{

#ifdef TRANSFORM_TEST
    if(primIdx == 0) transform_test();
#endif
#ifdef CALLABLE_TEST
    if(primIdx == 0) solve_callable_test();
#endif

    if(primIdx == 0)
    {
        unsigned partBuffer_size = partBuffer.size() ;
        unsigned planBuffer_size = planBuffer.size() ;
        unsigned tranBuffer_size = tranBuffer.size() ;

#ifdef WITH_PRINT
        rtPrintf("// intersect_analytic.cu:bounds buffer sizes pts:%4d pln:%4d trs:%4d (3x NumTVQ) \n", partBuffer_size, planBuffer_size, tranBuffer_size ); 
#endif
    }


    optix::Aabb* aabb = (optix::Aabb*)result;
    *aabb = optix::Aabb();

    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ; 


//#define WITH_PRINT_IDENTITY_BOUNDS 1 
#ifdef WITH_PRINT_IDENTITY_BOUNDS

    rtPrintf("// intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index %d instance_index %d primitive_count %3d primIdx %3d identity ( %7d %7d %7d %7d ) \n", 
       repeat_index, instance_index, primitive_count, primIdx, identity.x, identity.y, identity.z, identity.w );  

    unsigned instance_index_test = 10u ; 
    uint4 identity_test = identityBuffer[instance_index_test*primitive_count+primIdx] ; 
    rtPrintf("// intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index %d instance_index_test %d primitive_count %3d primIdx %3d identity ( %7d %7d %7d %7d ) \n", 
       repeat_index, instance_index_test, primitive_count, primIdx, identity_test.x, identity_test.y, identity_test.z, identity_test.w );  
 
#endif


    const Prim& prim    = primBuffer[primIdx];
 
    unsigned primFlag    = prim.primFlag() ;  
    unsigned partOffset  = prim.partOffset() ;  
    unsigned tranOffset  = prim.tranOffset() ; 
    unsigned numParts    = prim.numParts() ; 

//#define WITH_PRINT_PARTS 1
#ifdef WITH_PRINT_PARTS
    rtPrintf("// intersect_analysic.cu:bounds WITH_PRINT_PARTS repeat_index %d primIdx %d primFlag %d partOffset %d tranOffset %d numParts %d \n", 
       repeat_index, primIdx, primFlag, partOffset, tranOffset, numParts );
    for(unsigned p=0 ; p < numParts ; p++)
    {
        Part pt = partBuffer[partOffset + p] ; 
        unsigned typecode = pt.typecode() ; 
        unsigned boundary = pt.boundary() ; 
        rtPrintf("// intersect_analysic.cu:bounds WITH_PRINT_PARTS p %d typecode %d boundary %d \n", p, typecode, boundary );
    
    }
#endif



    if(primFlag == CSG_FLAGNODETREE || primFlag == CSG_FLAGINVISIBLE )  
    {
        // identity not strictly needed for bounds, but repeating whats done in intersect for debug convenience
        Part pt0 = partBuffer[partOffset + 0] ;
        unsigned typecode0 = pt0.typecode() ; 
        unsigned boundary0 = pt0.boundary() ;  

        csg_bounds_prim(primIdx, prim, aabb); 

#ifdef WITH_PRINT
        rtPrintf("// intersect_analytic.cu:bounds.NODETREE primIdx:%2d  bnd0:%3d typ0:%3d "
                 " min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", 
                    primIdx,
                    boundary0,
                    typecode0,
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4],
                    result[5]
                );
#endif

    }
#ifdef WITH_PARTLIST
    else if(primFlag == CSG_FLAGPARTLIST)  
    {

        for(unsigned int p=0 ; p < numParts ; p++)
        { 
            Part pt = partBuffer[partOffset + p] ; 
            unsigned typecode = pt.typecode() ; 
            unsigned boundary = pt.boundary() ; 

            identity.z = boundary ;  // boundary from partBuffer (see ggeo-/GPmt)

#ifdef WITH_PRINT
            rtPrintf("// intersect_analytic.cu:bounds.PARTLIST primIdx:%2d  p:%2d bnd:%3d typ:%3d pt.q2.f ( %10.4f %10.4f %10.4f %10.4f ) pt.q3.f ( %10.4f %10.4f %10.4f %10.4f ) \n", 
                    primIdx,
                    p,
                    boundary,
                    typecode,

                    pt.q2.f.x,
                    pt.q2.f.y,
                    pt.q2.f.z,
                    pt.q2.f.w,

                    pt.q3.f.x,
                    pt.q3.f.y,
                    pt.q3.f.z,
                    pt.q3.f.w
                    );
#endif

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
#endif
    else
    {
        rtPrintf("## intersect_analytic.cu:bounds ABORT BAD primflag %d \n", primFlag );
        return ; 
    }
    //rtPrintf("// intersect_analytic.cu:bounds primIdx %d primFlag %d partOffset %3d numParts %3d  min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", 
/*
    rtPrintf("// intersect_analytic.cu:bounds primIdx %d instance %2d id ( %3d %3d %3d %3d ) "
            " min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", 
        primIdx, 
        instance_index,
        identity.x, 
        identity.y, 
        identity.z, 
        identity.w, 
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5]
        );
*/

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

    if(primFlag == CSG_FLAGNODETREE)  
    { 
        evaluative_csg( prim, primIdx );
    }
    else if(primFlag == CSG_FLAGINVISIBLE)
    {
        // do nothing : report no intersections for primitives marked with primFlag CSG_FLAGINVISIBLE 
    }  
#ifdef WITH_PARTLIST
    else if(primFlag == CSG_FLAGPARTLIST)  
    {
        for(unsigned int p=0 ; p < numParts ; p++)
        {  
            Part pt = partBuffer[partOffset + p] ; 

            //identity.z = pt.boundary() ;   
            unsigned boundary = pt.boundary() ;   
            unsigned typecode = pt.typecode() ; 

            switch(typecode)
            {
                case CSG_ZERO:
                    intersect_aabb(pt.q2, pt.q3, boundary, primIdx);
                    break ; 
                case CSG_SPHERE:
                    intersect_zsphere<false>(pt.q0,pt.q1,pt.q2,pt.q3,boundary, primIdx);
                    break ; 
                case CSG_TUBS:
                    intersect_ztubs(pt.q0,pt.q1,pt.q2,pt.q3,boundary, primIdx);
                    break ; 
                case CSG_BOX:
                    intersect_box(pt.q0,boundary, primIdx);
                    break ; 
                case CSG_PRISM:
                    // q0.f param used in *bounds* to construct prismBuffer, which is used within intersect_prism
                    intersect_prism(boundary, primIdx);
                    break ; 
            }
        }
    } 
#endif
}


