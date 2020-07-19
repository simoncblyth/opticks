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



//#define CSG_BOUNDS_DEBUG 1


static __device__
void csg_bounds_prim(int primIdx, const Prim& prim, optix::Aabb* aabb )  // NB OptiX primitive, but can be CSG tree for Opticks
{
    unsigned partBuffer_size = partBuffer.size() ;
    unsigned planBuffer_size = planBuffer.size() ;
    unsigned tranBuffer_size = tranBuffer.size() ;

    const int partOffset = prim.partOffset();
    const int numParts   = prim.numParts() ;
    const int tranOffset = prim.tranOffset() ;
    const int planOffset = prim.planOffset() ;

    const unsigned primFlag   = prim.primFlag() ;  

    unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 

#ifdef CSG_BOUNDS_DEBUG
    unsigned numNodes = TREE_NODES(height) ;      

    if(primFlag != CSG_FLAGNODETREE)  
    {
        rtPrintf("## csg_bounds_prim ABORT expecting primFlag CSG_FLAGNODETREE \n");
        return ;  
    }

    rtPrintf("//csg_bounds_prim CSG_FLAGNODETREE "
             " primIdx %3d partOffset %3d "
             " numParts %3d -> height %2d -> numNodes %2d "
             " tranBuffer_size %3u "
             "\n", 
             primIdx, partOffset, 
             numParts, height, numNodes, 
             tranBuffer_size 
             );

#endif


    
    unsigned nodeIdx = 1 << height ; 
    while(nodeIdx)
    {
        int depth = TREE_DEPTH(nodeIdx) ;
        int elev = height - depth ; 

        Part pt = partBuffer[partOffset+nodeIdx-1];  // nodeIdx is 1-based

        unsigned typecode = pt.typecode() ; 
        unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
        bool complement = pt.complement() ; 


#ifdef CSG_BOUNDS_DEBUG
/*
        rtPrintf("//csg_bounds_prim "
                 " primIdx %3d nodeIdx %2u depth %2d elev %2d "
                 " typecode %2u tranOffset %2d gtransformIdx %2u complement %d" 
                 " \n", 
                  primIdx, nodeIdx, depth, elev, 
                  typecode, tranOffset, gtransformIdx, complement
                  );
*/
#endif

        if(gtransformIdx == 0)
        {
            switch(typecode)
            {
                case CSG_SPHERE:            csg_bounds_sphere(   pt.q0,               aabb, NULL ); break ;
                case CSG_ZSPHERE:           csg_bounds_zsphere(  pt.q0, pt.q1, pt.q2, aabb, NULL ); break ;
                case CSG_BOX:               csg_bounds_box(      pt.q0,               aabb, NULL ); break ;
                case CSG_BOX3:              csg_bounds_box3(     pt.q0,               aabb, NULL ); break ;
                case CSG_SLAB:              csg_bounds_slab(     pt.q0, pt.q1,        aabb, NULL ); break ; 
                case CSG_PLANE:             csg_bounds_plane(    pt.q0,               aabb, NULL ); break ;      
                case CSG_CYLINDER:          csg_bounds_cylinder( pt.q0, pt.q1,        aabb, NULL ); break ;  
                case CSG_DISC:              csg_bounds_disc(     pt.q0, pt.q1,        aabb, NULL ); break ;  
                case CSG_CONE:              csg_bounds_cone(     pt.q0,               aabb, NULL ); break ;  
#ifdef WITH_TORUS
                case CSG_TORUS:             csg_bounds_torus(    pt.q0,               aabb, NULL ); break ;  
#endif
#ifdef WITH_CUBIC
                case CSG_CUBIC:             csg_bounds_cubic(    pt.q0, pt.q1,        aabb, NULL ); break ;  
#endif
                case CSG_HYPERBOLOID:       csg_bounds_hyperboloid(    pt.q0,         aabb, NULL ); break ;  
                case CSG_CONVEXPOLYHEDRON:  csg_bounds_convexpolyhedron( pt,          aabb, NULL, planOffset ); break ;  
                default:                                                                     break ; 
            }
        }
        else
        {
            unsigned trIdx = 3*(tranOffset + gtransformIdx-1)+0 ;
            if(trIdx >= tranBuffer_size)
            { 
                rtPrintf("## bounds ABORT trIdx %3u overflows tranBuffer_size %3u \n", trIdx, tranBuffer_size );
                return ;  
            }
            optix::Matrix4x4 tr = tranBuffer[trIdx] ; 
            optix::Matrix4x4 vt = tranBuffer[trIdx+1] ;  // inverse transform 


/*
            rtPrintf("\n%8.3f %8.3f %8.3f %8.3f   (trIdx:%3d)[vt]\n%8.3f %8.3f %8.3f %8.3f\n", 
                  vt[0], vt[1], vt[2], vt[3], trIdx,
                  vt[4], vt[5], vt[6], vt[7]  
                 );  
            rtPrintf("\n%8.3f %8.3f %8.3f %8.3f   (trIdx:%3d)[vt]\n%8.3f %8.3f %8.3f %8.3f\n",
                     vt[8], vt[9], vt[10], vt[11], trIdx,
                    vt[12], vt[13], vt[14], vt[15]
                  );

*/

            switch(typecode)
            {
                case CSG_SPHERE:    csg_bounds_sphere(   pt.q0,               aabb, &tr ); break ;
                case CSG_ZSPHERE:   csg_bounds_zsphere(  pt.q0, pt.q1, pt.q2, aabb, &tr ); break ;
                case CSG_BOX:       csg_bounds_box(      pt.q0,               aabb, &tr ); break ;
                case CSG_BOX3:      csg_bounds_box3(     pt.q0,               aabb, &tr ); break ;
                case CSG_SLAB:      csg_bounds_slab(     pt.q0, pt.q1,        aabb, &tr ); break ; 
                case CSG_PLANE:     csg_bounds_plane(    pt.q0,               aabb, &tr ); break ; 
                case CSG_CYLINDER:  csg_bounds_cylinder( pt.q0, pt.q1,        aabb, &tr ); break ;     
                case CSG_DISC:      csg_bounds_disc(     pt.q0, pt.q1,        aabb, &tr ); break ;     
                case CSG_CONE:      csg_bounds_cone(     pt.q0,               aabb, &tr ); break ;
#ifdef WITH_TORUS
                case CSG_TORUS:     csg_bounds_torus(    pt.q0,               aabb, &tr ); break ;  
#endif
#ifdef WITH_CUBIC
                case CSG_CUBIC:     csg_bounds_cubic(    pt.q0, pt.q1,        aabb, &tr ); break ;  
#endif
                case CSG_HYPERBOLOID: csg_bounds_hyperboloid(    pt.q0,       aabb, &tr ); break ;  
                case CSG_CONVEXPOLYHEDRON:  csg_bounds_convexpolyhedron( pt,  aabb, &tr, planOffset ); break ;  
                default:                                                 break ; 
            }
        }
        nodeIdx = POSTORDER_NEXT( nodeIdx, elev ) ;
        // see opticks/dev/csg/postorder.py for explanation of bit-twiddling postorder  
    }
}



static __device__
void csg_intersect_part(const Prim& prim, const unsigned partIdx, const float& tt_min, float4& tt  )
{
    unsigned tranOffset = prim.tranOffset();
    unsigned planOffset = prim.planOffset();
    Part pt = partBuffer[partIdx] ; 

    unsigned typecode = pt.typecode() ; 
    unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
    bool complement = pt.complement();  

    bool valid_intersect = false ; 

    if(gtransformIdx == 0)
    {
        switch(typecode)
        {
            case CSG_SPHERE:    valid_intersect = csg_intersect_sphere(   pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_ZSPHERE:   valid_intersect = csg_intersect_zsphere(  pt.q0, pt.q1, pt.q2, tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_BOX:       valid_intersect = csg_intersect_box(      pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_BOX3:      valid_intersect = csg_intersect_box3(     pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_SLAB:      valid_intersect = csg_intersect_slab(     pt.q0, pt.q1,        tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_PLANE:     valid_intersect = csg_intersect_plane(    pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_CYLINDER:  valid_intersect = csg_intersect_cylinder( pt.q0, pt.q1,        tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_DISC:      valid_intersect = csg_intersect_disc(     pt.q0, pt.q1,        tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_CONE:      valid_intersect = csg_intersect_cone(     pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
#ifdef WITH_TORUS
            case CSG_TORUS:            valid_intersect = csg_intersect_torus(          pt.q0,  tt_min, tt, ray.origin, ray.direction ) ; break ; 
#endif
#ifdef WITH_CUBIC
            case CSG_CUBIC:            valid_intersect = csg_intersect_cubic( pt.q0, pt.q1,    tt_min, tt, ray.origin, ray.direction ) ; break ; 
#endif
            case CSG_HYPERBOLOID:      valid_intersect = csg_intersect_hyperboloid(    pt.q0,  tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_CONVEXPOLYHEDRON: valid_intersect = csg_intersect_convexpolyhedron( pt,   tt_min, tt, ray.origin, ray.direction, planOffset ) ; break ; 
        }
    }
    else
    {
        unsigned tIdx = 3*(tranOffset + gtransformIdx-1) ;  // transform
        if(tIdx + 2 >= tranBuffer.size())
        { 
            rtPrintf("##csg_intersect_part ABORT tIdx+2 %3u overflows tranBuffer.size \n", tIdx+2 );
            return ;  
        }

        //optix::Matrix4x4 T = tranBuffer[tIdx+0] ;  // transform (not used here, but needed in bbox transforming in bounds)
        optix::Matrix4x4 V = tranBuffer[tIdx+1] ;  // inverse transform 
        optix::Matrix4x4 Q = tranBuffer[tIdx+2] ;  // inverse transform transposed

        float4 origin    = make_float4( ray.origin.x, ray.origin.y, ray.origin.z, 1.f );           // w=1 for position  
        float4 direction = make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0.f );  // w=0 for vector

        origin    = origin * V ;    // world frame into primitive frame with inverse transform
        direction = direction * V ;  // <-- will loose normalization with scaling, intersects MUST NOT assume normalized ray direction

        float3 ray_origin = make_float3( origin.x, origin.y, origin.z );
        float3 ray_direction = make_float3( direction.x, direction.y, direction.z ); 

        switch(typecode)
        {
            case CSG_SPHERE:    valid_intersect = csg_intersect_sphere(   pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_ZSPHERE:   valid_intersect = csg_intersect_zsphere(  pt.q0, pt.q1, pt.q2, tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_BOX:       valid_intersect = csg_intersect_box(      pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_BOX3:      valid_intersect = csg_intersect_box3(     pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_SLAB:      valid_intersect = csg_intersect_slab(     pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_PLANE:     valid_intersect = csg_intersect_plane(    pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_CYLINDER:  valid_intersect = csg_intersect_cylinder( pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_DISC:      valid_intersect = csg_intersect_disc(     pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_CONE:      valid_intersect = csg_intersect_cone(     pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
#ifdef WITH_TORUS
            case CSG_TORUS:     valid_intersect = csg_intersect_torus(    pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
#endif
#ifdef WITH_CUBIC
            case CSG_CUBIC:     valid_intersect = csg_intersect_cubic(    pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
#endif
            case CSG_HYPERBOLOID:      valid_intersect = csg_intersect_hyperboloid(    pt.q0,  tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_CONVEXPOLYHEDRON: valid_intersect = csg_intersect_convexpolyhedron( pt,   tt_min, tt, ray_origin, ray_direction, planOffset ) ; break ; 
        }

        if(valid_intersect)
        {
            float4 ttn = make_float4( tt.x, tt.y, tt.z , 0.f );

            ttn = ttn * Q   ;  // primitive frame normal into world frame, using inverse transform transposed
            // TODO: try "V * ttn" as way to avoid the Q ???    
        
            tt.x = ttn.x ; 
            tt.y = ttn.y ; 
            tt.z = ttn.z ; 
        }
    }

    if(complement)  // flip normal, even for miss need to signal the complement with a -0.f  
    {
        // For valid_intersects this flips the normal
        // otherwise for misses all tt.xyz values should be zero
        // but nevertheless proceed to set signbits to signal a complement miss  
        // to the caller... csg_intersect_boolean

        tt.x = -tt.x ; 
        tt.y = -tt.y ; 
        tt.z = -tt.z ; 
    }

}



