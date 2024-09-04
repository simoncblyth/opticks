#pragma once

LEAF_FUNC
float distance_leaf_convexpolyhedron( const float3& pos, const CSGNode* node, const float4* plan )
{
    unsigned planeIdx = node->planeIdx() ; 
    unsigned planeNum = node->planeNum() ; 
    float sd = 0.f ; 
    for(unsigned i=0 ; i < planeNum ; i++) 
    {    
        const float4& plane = plan[planeIdx+i];   
        float d = plane.w ;
        float3 n = make_float3(plane);
        float sd_plane = dot(pos, n) - d ; 
        sd = i == 0 ? sd_plane : fmaxf( sd, sd_plane ); 
    }
    return sd ; 
}


LEAF_FUNC
void intersect_leaf_convexpolyhedron( bool& valid_isect, float4& isect, const CSGNode* node, const float4* plan, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float t0 = -CUDART_INF_F ; 
    float t1 =  CUDART_INF_F ; 

    float3 t0_normal = make_float3(0.f);
    float3 t1_normal = make_float3(0.f);

    unsigned planeIdx = node->planeIdx() ; 
    unsigned planeNum = node->planeNum() ; 

    for(unsigned i=0 ; i < planeNum ; i++) 
    {    
        const float4& plane = plan[planeIdx+i];   
        float3 n = make_float3(plane);
        float dplane = plane.w ;

         // RTCD p199,  
         //            n.X = dplane
         //   
         //             n.(o+td) = dplane
         //            no + t nd = dplane
         //                    t = (dplane - no)/nd
         //   

        float nd = dot(n, ray_direction); // -ve: entering, +ve exiting halfspace  
        float no = dot(n, ray_origin ) ;  //  distance from coordinate origin to ray origin in direction of plane normal 
        float dist = no - dplane ;        //  subtract plane distance from origin to get signed distance from plane, -ve inside 
        float t_cand = -dist/nd ;

        bool parallel_inside = nd == 0.f && dist < 0.f ;   // ray parallel to plane and inside halfspace
        bool parallel_outside = nd == 0.f && dist > 0.f ;  // ray parallel to plane and outside halfspac

        if(parallel_inside) continue ;       // continue to next plane 
        if(parallel_outside)
        {
            valid_isect = false ;   
            return ;  // <-- without early exit, this still works due to infinity handling 
        }

        //    NB ray parallel to plane and outside halfspace 
        //         ->  t_cand = -inf 
        //                 nd = 0.f 
        //                t1 -> -inf  

        if( nd < 0.f)  // entering 
        {
            if(t_cand > t0)
            {
                t0 = t_cand ;
                t0_normal = n ;
            }
        }
        else     // exiting
        {
            if(t_cand < t1)
            {
                t1 = t_cand ;
                t1_normal = n ;
            }
        }
    }

    valid_isect = t0 < t1 ;
    if(valid_isect)
    {
        if( t0 > t_min )
        {
            isect.x = t0_normal.x ;
            isect.y = t0_normal.y ;
            isect.z = t0_normal.z ;
            isect.w = t0 ;
        }
        else if( t1 > t_min )
        {
            isect.x = t1_normal.x ;
            isect.y = t1_normal.y ;
            isect.z = t1_normal.z ;
            isect.w = t1 ;
        }
    }
}



