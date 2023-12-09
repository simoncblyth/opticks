#pragma once

LEAF_FUNC
float distance_leaf_zsphere(const float3& pos, const quad& q0, const quad& q1 )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;
    const float2 zdelta = make_float2(q1.f);
    const float z2 = center.z + zdelta.y ; 
    const float z1 = center.z + zdelta.x ;    

    float3 p = pos - center;
    float sd_sphere = length(p) - radius ; 
    float sd_capslab = fmaxf( pos.z - z2 , z1 - pos.z ); 

    float sd = fmaxf( sd_capslab, sd_sphere );    // CSG intersect
    return sd ; 
}

/**
intersect_leaf_zsphere
------------------------

HMM: rays that look destined to land near to "apex" have a rare (order 1 in 300k) 
problem of missing the zsphere.  This is probably arising from the upper cap 
implementation acting effectively like cutting a pinhole at the apex. 

When there is no upper cap perhaps can avoid the problem by setting zmax to beyond the 
apex ? Or could have a different imp for zsphere with lower cap but no upper cap. 

Note that zsphere with no upper cap is used a lot for PMTs so a simpler imp
for zsphere without upper cut does make sense.  

NB "z2sph <= zmax" changed from "z2sph < zmax" Aug 29, 2022

The old inequality caused rare unexpected MISS for rays that would
have been expected to intersect close to the apex of the zsphere  

See : notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst

**/


LEAF_FUNC
bool intersect_leaf_zsphere(float4& isect, const quad& q0, const quad& q1, const float& t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float3 center = make_float3(q0.f);
    float3 O = ray_origin - center;  
    float3 D = ray_direction;
    const float radius = q0.f.w;

    float b = dot(O, D);               // t of closest approach to sphere center
    float c = dot(O, O)-radius*radius; // < 0. indicates ray_origin inside sphere

#ifdef DEBUG_RECORD
    printf("//[intersect_leaf_zsphere radius %10.4f b %10.4f c %10.4f \n", radius, b, c); 
#endif

    if( c > 0.f && b > 0.f ) return false ;    
    // Cannot intersect when ray origin outside sphere and direction away from sphere.
    // Whether early exit speeds things up is another question ... 

    const float2 zdelta = make_float2(q1.f);
    const float zmax = center.z + zdelta.y ;   // + 0.1f artificial increase zmax to test apex bug 
    const float zmin = center.z + zdelta.x ;    

#ifdef DEBUG_RECORD
    bool with_upper_cut = zmax < radius ; 
    bool with_lower_cut = zmin > -radius ; 
    printf("// intersect_leaf_zsphere radius %10.4f zmax %10.4f zmin %10.4f  with_upper_cut %d with_lower_cut %d  \n", radius, zmax, zmin, with_upper_cut, with_lower_cut ); 
#endif


    float d = dot(D, D);               // NB NOT assuming normalized ray_direction

    float t1sph, t2sph, disc, sdisc ;    
    robust_quadratic_roots(t1sph, t2sph, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

    float z1sph = ray_origin.z + t1sph*ray_direction.z ;  // sphere z intersects
    float z2sph = ray_origin.z + t2sph*ray_direction.z ; 

#ifdef DEBUG_RECORD
    printf("// intersect_leaf_zsphere t1sph %10.4f t2sph %10.4f sdisc %10.4f \n", t1sph, t2sph, sdisc ); 
    printf("// intersect_leaf_zsphere z1sph %10.4f z2sph %10.4f zmax %10.4f zmin %10.4f sdisc %10.4f \n", z1sph, z2sph, zmax, zmin, sdisc ); 
#endif

    float idz = 1.f/ray_direction.z ; 
    float t_QCAP = (zmax - ray_origin.z)*idz ;   // upper cap intersects
    float t_PCAP = (zmin - ray_origin.z)*idz ;   // lower cap intersect 


    float t1cap = fminf( t_QCAP, t_PCAP ) ;      // order cap intersects along the ray 
    float t2cap = fmaxf( t_QCAP, t_PCAP ) ;      // t2cap > t1cap 

#ifdef DEBUG_RECORD
    bool t1cap_disqualify = t1cap < t1sph || t1cap > t2sph ; 
    bool t2cap_disqualify = t2cap < t1sph || t2cap > t2sph ;  
    printf("//intersect_leaf_zsphere t1sph %7.3f t2sph %7.3f t_QCAP %7.3f t_PCAP %7.3f t1cap %7.3f t2cap %7.3f  \n", t1sph, t2sph, t_QCAP, t_PCAP, t1cap, t2cap ); 
    printf("//intersect_leaf_zsphere  t1cap_disqualify %d t2cap_disqualify %d \n", t1cap_disqualify, t2cap_disqualify  ); 
#endif

    // disqualify plane intersects outside sphere t range
    if(t1cap < t1sph || t1cap > t2sph) t1cap = t_min ; 
    if(t2cap < t1sph || t2cap > t2sph) t2cap = t_min ; 

    // hmm somehow is seems unclean to have to use both z and t language

    float t_cand = t_min ; 
    if(sdisc > 0.f)
    {

#ifdef DEBUG_RECORD
        //std::raise(SIGINT); 
#endif

        if(      t1sph > t_min && z1sph > zmin && z1sph <= zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
        else if( t1cap > t_min )                                   t_cand = t1cap ;  // t1cap qualifies -> t1cap 
        else if( t2cap > t_min )                                   t_cand = t2cap ;  // t2cap qualifies -> t2cap
        else if( t2sph > t_min && z2sph > zmin && z2sph <= zmax)   t_cand = t2sph ;  // t2sph qualifies and t2cap disabled or disqialified -> t2sph
    }

    bool valid_isect = t_cand > t_min ;
#ifdef DEBUG_RECORD
    printf("//intersect_leaf_zsphere valid_isect %d t_min %7.3f t1sph %7.3f t1cap %7.3f t2cap %7.3f t2sph %7.3f t_cand %7.3f \n", valid_isect, t_min, t1sph, t1cap, t2cap, t2sph, t_cand ); 
#endif

    if(valid_isect)
    {
        isect.w = t_cand ;
        if( t_cand == t1sph || t_cand == t2sph)
        {
            isect.x = (O.x + t_cand*D.x)/radius ; // normalized by construction
            isect.y = (O.y + t_cand*D.y)/radius ;
            isect.z = (O.z + t_cand*D.z)/radius ;
        }
        else
        {
            isect.x = 0.f ;
            isect.y = 0.f ;
            isect.z = t_cand == t_PCAP ? -1.f : 1.f ;
        }
    }

#ifdef DEBUG_RECORD
    printf("//]intersect_leaf_zsphere valid_isect %d \n", valid_isect ); 
#endif
    return valid_isect ;
}


