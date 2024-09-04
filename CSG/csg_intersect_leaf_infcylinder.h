#pragma once
/**
intersect_leaf_infcylinder
----------------------------------

Use standard Z-axial cylinder orientation to see how much it simplifies::

    x^2 + y^2 = r^2

    (Ox + t Dx)^2 + (Oy + t Dy)^2 = r^2 

    Ox ^2 + t^2 Dx^2 + 2 t Ox Dx   
    Oy ^2 + t^2 Dy^2 + 2 t Oy Dy


    t^2 (Dx^2 + Dy^2) + 2 t ( OxDx + Oy Dy ) + Ox^2 + Oy^2 - r^2  = 0     

Contrast this eqn with that on RTCD p195 "bk-;bk-rtcd 195"  : its a natural simplification.
Instead of dotting all components and subtracting the axial part can just directly 
dot the non-axial x and y thanks to the fixed orientation.

**/

LEAF_FUNC
void intersect_leaf_infcylinder( bool& valid_isect, float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float r = q0.f.w ; 

    const float3& O = ray_origin ;    
    const float3& D = ray_direction ;    

    float a = D.x*D.x + D.y*D.y ; 
    float b = O.x*D.x + O.y*D.y ; 
    float c = O.x*O.x + O.y*O.y - r*r ;  

    float disc = b*b-a*c;

    if(disc > 0.0f)  // has intersections with the infinite cylinder
    {
        float t_NEAR, t_FAR, sdisc ;   

        robust_quadratic_roots(t_NEAR, t_FAR, disc, sdisc, a, b, c); //  Solving:  a t^2 + 2 b t +  c = 0 

        float t_cand = sdisc > 0.f ? ( t_NEAR > t_min ? t_NEAR : t_FAR ) : t_min ;

        valid_isect = t_cand > t_min ; 

        if( valid_isect  )
        {
            isect.x = (O.x + t_cand*D.x)/r ;   // normalized by construction
            isect.y = (O.y + t_cand*D.y)/r ;
            isect.z = 0.f ;
            isect.w = t_cand ; 
        }
    }
}

