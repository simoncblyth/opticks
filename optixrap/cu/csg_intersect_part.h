



static __device__
void csg_intersect_part(unsigned partIdx, const float& tt_min, float4& tt  )
{
    Part pt = partBuffer[partIdx] ; 
    unsigned typecode = pt.typecode() ; 
    unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None

    if(gtransformIdx == 0)
    {
        switch(typecode)
        {
            case CSG_SPHERE: csg_intersect_sphere(pt.q0,tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_BOX:    csg_intersect_box(   pt.q0, tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_SLAB:   csg_intersect_slab(  pt.q0,pt.q1, tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_PLANE:  csg_intersect_plane( pt.q0, tt_min, tt, ray.origin, ray.direction )        ; break ; 
            case CSG_CYLINDER:  csg_intersect_cylinder( pt.q0, pt.q1, tt_min, tt, ray.origin, ray.direction )        ; break ; 
        }
    }
    else
    {
        unsigned tIdx = 3*(gtransformIdx-1) ;  // transform
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
            case CSG_SPHERE: valid_intersect = csg_intersect_sphere(pt.q0,tt_min, tt, ray_origin, ray_direction )  ; break ; 
            case CSG_BOX:    valid_intersect = csg_intersect_box(   pt.q0,tt_min, tt, ray_origin, ray_direction )  ; break ; 
            case CSG_SLAB:   valid_intersect = csg_intersect_slab(  pt.q0,pt.q1, tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_PLANE:  valid_intersect = csg_intersect_plane( pt.q0, tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_CYLINDER:  valid_intersect = csg_intersect_cylinder( pt.q0, pt.q1, tt_min, tt, ray.origin, ray.direction ) ; break ; 
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






