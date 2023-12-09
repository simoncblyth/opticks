#pragma once

/**
intersect_leaf_disc
---------------------

RTCD p197  (Real Time Collision Detection)

CSG_DISC was implemented to avoid degeneracy/speckle problems when using CSG_CYLINDER
to describe very flat cylinders such as Daya Bays ESR mirror surface. 
Note that the simplicity of disc intersects compared to cylinder has allowed 
inner radius handling (in param.f.z) for easy annulus definition without using CSG subtraction.

NB ray-plane intersects are performed with the center disc only at:  z = zc = (z1+z2)/2 
The t_center obtained is then deltared up and down depending on (z2-z1)/2

This approach appears to avoid the numerical instability speckling problems encountered 
with csg_intersect_cylinder when dealing with very flat disc like cylinders. 

Note that intersects with the edge of the disk are not implemented, if such intersects
are relevant you need to use CSG_CYLINDER not CSG_DISC.


For testing see tboolean-esr and tboolean-disc.::

                r(t) = O + t n 

                               ^ /         ^ 
                               |/          | d
         ----------------------+-----------|-------------------------------- z2
                              /            |
         - - - - - - - - - - * - - - - - - C- - -  - - - - - - - - - - - - - zc
                            /
         ------------------+------------------------------------------------ z1
                          /|
                         / V
                        /
                       O

          m = O - C


To work as a CSG sub-object MUST have a different intersect 
on the other side and normals must be rigidly attached to 
geometry (must not depend on ray direction)


Intersect of ray and plane::

    r(t) = ray_origin + t * ray_direction

    (r(t) - center).d  = ( m + t * n ).d  = 0    <-- at intersections of ray and plane thru center with normal d 

    t = -m.d / n.d 

Consider wiggling center up to z2 and down to z1 (in direction of normal d) n.d is unchanged::

    (r(t) - (center+ delta d )).d = 0

    (m - delta d ).d + t * n.d = 0 

    m.d - delta + t* nd = 0 

    t =  -(m.d + delta) / n.d              

      = -m.d/n.d  +- delta/n.d


Intersect is inside disc radius when::

    rsq =   (r(t) - center).(r(t) - center) < radius*radius

    (m + t n).(m + t n)  <  rr

    t*t nn + 2 t nm + mm  <  rr  

    t ( 2 nm + t nn ) + mm   <  rr    

    rsq < rr    checkr(from cylinder) is: rsq - rr 


Determine whether the t_cand intersect hit after delta-ing 
is on the upside (normal +Z) or downside (normal -Z) of disc
from the sign of the below dot product, allowing determination 
of the rigid outward normal direction.::

    r(t) = ray_origin + t * ray_direction

    (r(t_cand) - center).d  = m.d + t_cand n.d     

**/

LEAF_FUNC
bool intersect_leaf_disc(float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float   inner  = q0.f.z ; 
    const float   radius = q0.f.w ; 
    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ;            // NB z2 > z1 by assertion in npy-/NDisc.cpp
    const float       zc = (z1 + z2)/2.f  ;     // avg
    const float       zdelta = (z2 - z1)/2.f ;  // +ve half difference 

    const float3 center = make_float3( q0.f.x, q0.f.y, zc ); // C: point at middle of disc

#ifdef DEBUG
    printf("//intersect_leaf_disc (%10.4f, %10.4f, %10.4f) \n", center.x, center.y, center.z ); 
#endif


    const float3 m = ray_origin - center ;            // m: ray origin in disc frame
    const float3 n = ray_direction ;                  // n: ray direction vector (not normalized)
    const float3 d = make_float3(0.f, 0.f, 1.f );     // d: normal to the disc (normalized)

    float rr = radius*radius ; 
    float ii = inner*inner ; 

    float mm = dot(m, m) ; 
    float nn = dot(n, n) ; 
    float nd = dot(n, d) ;   // >0 : ray direction in same hemi as normal
    float md = dot(m, d) ;
    float mn = dot(m, n) ; 

    float t_center = -md/nd ; 
    float rsq = t_center*(2.f*mn + t_center*nn) + mm  ;   // ( m + tn).(m + tn) 

    float t_delta  = nd < 0.f ? -zdelta/nd : zdelta/nd ;    // <-- pragmatic make t_delta +ve

    float root1 = t_center - t_delta ; 
    float root2 = t_center + t_delta ;   // root2 > root1
 
    float t_cand = ( rsq < rr && rsq > ii ) ? ( root1 > t_min ? root1 : root2 ) : t_min ; 

    float side = md + t_cand*nd ;    

    bool valid_isect = t_cand > t_min ;
    if(valid_isect)
    {        
        isect.x = 0.f ; 
        isect.y = 0.f ; 
        isect.z = side > 0.f ? 1.f : -1.f ; 
        isect.w = t_cand  ; 
    }
    return valid_isect ; 
}


