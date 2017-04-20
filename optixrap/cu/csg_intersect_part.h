static __device__
void csg_bounds_prim(const Prim& prim, optix::Aabb* aabb )
{
    unsigned tranBuffer_size = tranBuffer.size() ;

    const unsigned partOffset = prim.partOffset();
    const unsigned numParts   = prim.numParts() ;
    const unsigned tranOffset = prim.tranOffset() ;
    const unsigned primFlag   = prim.primFlag() ;  


    if(primFlag != CSG_FLAGNODETREE)  
    {
        rtPrintf("## csg_bounds_prim ABORT \n");
        return ;  
    }

    unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
    unsigned numNodes = TREE_NODES(height) ;      

    rtPrintf("##csg_bounds_prim partOffset %2d numParts %2d height %2d numNodes %2d tranBuffer_size %3u \n", partOffset, numParts, height, numNodes, tranBuffer_size );

    uint4 identity = identityBuffer[instance_index] ;  // instance_index from OGeo is 0 for non-instanced

    
    unsigned nodeIdx = 1 << height ; 
    while(nodeIdx)
    {
        int depth = TREE_DEPTH(nodeIdx) ;
        int elev = height - depth ; 

        Part pt = partBuffer[partOffset+nodeIdx-1];  // nodeIdx is 1-based

        unsigned typecode = pt.typecode() ; 
        unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
    
        rtPrintf("##csg_bounds_prim nodeIdx %2u depth %2d elev %2d typecode %2u gtransformIdx %2u \n", nodeIdx, depth, elev, typecode, gtransformIdx );

        if(gtransformIdx == 0)
        {
            switch(typecode)
            {
                case CSG_SPHERE:     csg_bounds_sphere(   pt.q0,               aabb, NULL ); break ;
                case CSG_ZSPHERE:    csg_bounds_zsphere(  pt.q0, pt.q1, pt.q2, aabb, NULL ); break ;
                case CSG_BOX:        csg_bounds_box(      pt.q0,               aabb, NULL ); break ;
                case CSG_SLAB:       csg_bounds_slab(     pt.q0, pt.q1,        aabb, NULL ); break ; 
                case CSG_PLANE:      csg_bounds_plane(    pt.q0,               aabb, NULL ); break ;      
                case CSG_CYLINDER:   csg_bounds_cylinder( pt.q0, pt.q1,        aabb, NULL ); break ;  
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
            switch(typecode)
            {
                case CSG_SPHERE:    csg_bounds_sphere(   pt.q0,               aabb, &tr ); break ;
                case CSG_ZSPHERE:   csg_bounds_zsphere(  pt.q0, pt.q1, pt.q2, aabb, &tr ); break ;
                case CSG_BOX:       csg_bounds_box(      pt.q0,               aabb, &tr ); break ;
                case CSG_SLAB:      csg_bounds_slab(     pt.q0, pt.q1,        aabb, &tr ); break ; 
                case CSG_PLANE:     csg_bounds_plane(    pt.q0,               aabb, &tr ); break ; 
                case CSG_CYLINDER:  csg_bounds_cylinder( pt.q0, pt.q1,        aabb, &tr ); break ;     
                default:                                                 break ; 
            }
        }
        nodeIdx = POSTORDER_NEXT( nodeIdx, elev ) ;
        // see opticks/dev/csg/postorder.py for explanation of bit-twiddling postorder  
    }
}



static __device__
void csg_intersect_part(const unsigned tranOffset, const unsigned partIdx, const float& tt_min, float4& tt  )
{
    Part pt = partBuffer[partIdx] ; 

    unsigned typecode = pt.typecode() ; 
    unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None

    if(gtransformIdx == 0)
    {
        switch(typecode)
        {
            case CSG_SPHERE:    csg_intersect_sphere(   pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_ZSPHERE:   csg_intersect_zsphere(  pt.q0, pt.q1, pt.q2, tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_BOX:       csg_intersect_box(      pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_SLAB:      csg_intersect_slab(     pt.q0, pt.q1,        tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_PLANE:     csg_intersect_plane(    pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ; 
            case CSG_CYLINDER:  csg_intersect_cylinder( pt.q0, pt.q1,        tt_min, tt, ray.origin, ray.direction ) ; break ; 
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

        bool valid_intersect = false ; 

        switch(typecode)
        {
            case CSG_SPHERE:    valid_intersect = csg_intersect_sphere(   pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_ZSPHERE:   valid_intersect = csg_intersect_zsphere(  pt.q0, pt.q1, pt.q2, tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_BOX:       valid_intersect = csg_intersect_box(      pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_SLAB:      valid_intersect = csg_intersect_slab(     pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_PLANE:     valid_intersect = csg_intersect_plane(    pt.q0,               tt_min, tt, ray_origin, ray_direction ) ; break ; 
            case CSG_CYLINDER:  valid_intersect = csg_intersect_cylinder( pt.q0, pt.q1,        tt_min, tt, ray_origin, ray_direction ) ; break ; 
        }

        if(valid_intersect)
        {
            float4 ttn = make_float4( tt.x, tt.y, tt.z , 0.f );

            ttn = ttn * Q   ;  // primitive frame normal into world frame, using inverse transform transposed
            
            tt.x = ttn.x ; 
            tt.y = ttn.y ; 
            tt.z = ttn.z ; 
        }
    }
}



