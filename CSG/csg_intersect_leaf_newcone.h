#pragma once
/**
intersect_leaf_newcone
------------------------

Old cone incorrectly assumed that rays would always intersect the infinite cone.


**fminf(float4) picking t_cand implications**

1. cannot allow t_min disqualified t into roots

   * BUT that check can be combined with z-range or r-range qualifications

2. must use RT_DEFAULT_MAX to represent disqualified, not t_min as is often used


**normal**

x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 

Gradient:    [2x, 2y, (-2z + 2z0) tanth^2 ] 
Gradient:    2*[x, y, (z0-z) tanth^2 ] 

huh : is there a simpler way to get normal ? just from cone param ?


**References**

* https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf

**/


LEAF_FUNC
bool intersect_leaf_newcone( float4& isect, const quad& q0, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    const float& r1 = q0.f.x ; 
    const float& z1 = q0.f.y ; 
    const float& r2 = q0.f.z ; 
    const float& z2 = q0.f.w ;   // z2 > z1

    const float r1r1 = r1*r1 ; 
    const float r2r2 = r2*r2 ; 
    const float tth = (r2-r1)/(z2-z1) ;
    const float tth2 = tth*tth ; 
    const float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex

    const float3& o = ray_origin ;
    const float3& d = ray_direction ;
    const float idz = 1.f/d.z ;  

#ifdef DEBUG_CONE
    printf("//intersect_leaf_newcone r1 %10.4f z1 %10.4f r2 %10.4f z2 %10.4f : z0 %10.4f \n", r1, z1, r2, z2, z0 );  
#endif
 
    // intersects with cap planes
    float t_cap1 = d.z == 0.f ? RT_DEFAULT_MAX : (z1 - o.z)*idz ;   // d.z zero means no z-plane intersects
    float t_cap2 = d.z == 0.f ? RT_DEFAULT_MAX : (z2 - o.z)*idz ;  // HMM: could just let the infinities arise ?
    // radii squared at cap intersects  
    const float rr_cap1 = (o.x + t_cap1*d.x)*(o.x + t_cap1*d.x) + (o.y + t_cap1*d.y)*(o.y + t_cap1*d.y) ;  
    const float rr_cap2 = (o.x + t_cap2*d.x)*(o.x + t_cap2*d.x) + (o.y + t_cap2*d.y)*(o.y + t_cap2*d.y) ;  

    t_cap1 = rr_cap1 < r1r1 && t_cap1 > t_min ? t_cap1 : RT_DEFAULT_MAX ;  // disqualify out-of-radius
    t_cap2 = rr_cap2 < r2r2 && t_cap2 > t_min ? t_cap2 : RT_DEFAULT_MAX ; 
 
    // collecting terms to form coefficients of the quadratic : c2 t^2 + 2 c1 t + c0 = 0 
    const float c2 = d.x*d.x + d.y*d.y - d.z*d.z*tth2 ;
    const float c1 = o.x*d.x + o.y*d.y - (o.z-z0)*d.z*tth2 ; 
    const float c0 = o.x*o.x + o.y*o.y - (o.z-z0)*(o.z-z0)*tth2 ;

    float t_near, t_far, disc, sdisc ;   
    robust_quadratic_roots_disqualifying(RT_DEFAULT_MAX, t_near, t_far, disc, sdisc, c2, c1, c0 ) ;


#ifdef DEBUG_CONE
    printf("//intersect_leaf_newcone c2 %10.4f c1 %10.4f c0 %10.4f disc %10.4f disc > 0.f %d : tth %10.4f \n", c2, c1, c0, disc, disc>0.f, tth  );  
#endif

    const float z_near = o.z+t_near*d.z ; 
    const float z_far  = o.z+t_far*d.z ; 

    t_near = z_near > z1 && z_near < z2  && t_near > t_min ? t_near : RT_DEFAULT_MAX ; // disqualify out-of-z
    t_far  = z_far  > z1 && z_far  < z2  && t_far  > t_min ? t_far  : RT_DEFAULT_MAX ; 

    const float4 roots = make_float4( t_near, t_far, t_cap1, t_cap2 );
    const float t_cand = fminf(roots) ; 
    
    bool valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
    if(valid_isect)
    {
        if( t_cand == t_cap1 || t_cand == t_cap2 )
        {
            isect.x = 0.f ; 
            isect.y = 0.f ;
            isect.z = t_cand == t_cap2 ? 1.f : -1.f  ;   
        }
        else
        { 
            float3 n = normalize(make_float3( o.x+t_cand*d.x, o.y+t_cand*d.y, (z0-(o.z+t_cand*d.z))*tth2  ))  ; 
            isect.x = n.x ; 
            isect.y = n.y ;
            isect.z = n.z ; 
        }
        isect.w = t_cand ; 
    }
    return valid_isect ; 
}


