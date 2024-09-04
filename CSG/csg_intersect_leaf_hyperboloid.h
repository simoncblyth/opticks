#pragma once

/**
intersect_leaf_hyperboloid
-----------------------------

* http://mathworld.wolfram.com/One-SheetedHyperboloid.html

::

      x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )
      x^2 + y^2 - (r0^2/zf^2) * z^2 - r0^2  =  0 
      x^2 + y^2 + A * z^2 + B   =  0 

      grad( x^2 + y^2 + A * z^2 + B ) =  [2 x, 2 y, A*2z ] 


     (ox+t sx)^2 + (oy + t sy)^2 + A (oz+ t sz)^2 + B = 0 

      t^2 ( sxsx + sysy + A szsz ) + 2*t ( oxsx + oysy + A * ozsz ) +  (oxox + oyoy + A * ozoz + B ) = 0 

**/

LEAF_FUNC
void intersect_leaf_hyperboloid(bool& valid_isect, float4& isect, const quad& q0, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float zero(0.f); 
    const float one(1.f); 

    const float r0 = q0.f.x ;  // waist (z=0) radius 
    const float zf = q0.f.y ;  // at z=zf radius grows to  sqrt(2)*r0 
    const float z1 = q0.f.z ;  // z1 < z2 by assertion  
    const float z2 = q0.f.w ;  

    const float rr0 = r0*r0 ;
    const float z1s = z1/zf ; 
    const float z2s = z2/zf ; 
    const float rr1 = rr0 * ( z1s*z1s + one ) ; // radii squared at z=z1, z=z2
    const float rr2 = rr0 * ( z2s*z2s + one ) ;

    const float A = -rr0/(zf*zf) ;
    const float B = -rr0 ;  

    const float& sx = ray_direction.x ; 
    const float& sy = ray_direction.y ; 
    const float& sz = ray_direction.z ;

    const float& ox = ray_origin.x ; 
    const float& oy = ray_origin.y ; 
    const float& oz = ray_origin.z ;

    const float d = sx*sx + sy*sy + A*sz*sz ; 
    const float b = ox*sx + oy*sy + A*oz*sz ; 
    const float c = ox*ox + oy*oy + A*oz*oz + B ; 
    
    float t1hyp, t2hyp, disc, sdisc ;   
    robust_quadratic_roots(t1hyp, t2hyp, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

    const float h1z = oz + t1hyp*sz ;  // hyp intersect z positions
    const float h2z = oz + t2hyp*sz ; 

    //  z = oz+t*sz -> t = (z - oz)/sz 
    float osz = one/sz ; 
    float t2cap = (z2 - oz)*osz ;   // cap plane intersects
    float t1cap = (z1 - oz)*osz ;

    const float3 c1 = ray_origin + t1cap*ray_direction ; 
    const float3 c2 = ray_origin + t2cap*ray_direction ; 

    float crr1 = c1.x*c1.x + c1.y*c1.y ;   // radii squared at cap plane intersects
    float crr2 = c2.x*c2.x + c2.y*c2.y ; 

    // NB must disqualify t < t_min at "front" and "back" 
    // as this potentially picks between hyp intersects eg whilst near(t_min) scanning  

    const float4 t_cand_ = make_float4(   // restrict radii of cap intersects and z of hyp intersects
                                          t1hyp > t_min && disc > zero && h1z > z1 && h1z < z2 ? t1hyp : RT_DEFAULT_MAX ,
                                          t2hyp > t_min && disc > zero && h2z > z1 && h2z < z2 ? t2hyp : RT_DEFAULT_MAX ,
                                          t2cap > t_min && crr2 < rr2                          ? t2cap : RT_DEFAULT_MAX ,
                                          t1cap > t_min && crr1 < rr1                          ? t1cap : RT_DEFAULT_MAX 
                                      ) ;

    float t_cand = fminf( t_cand_ );  

    valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
    if(valid_isect)
    {        
        isect.w = t_cand ; 
        if( t_cand == t1hyp || t_cand == t2hyp )
        {
            const float3 p = ray_origin + t_cand*ray_direction ; 
            float3 n = normalize(make_float3( p.x,  p.y,  A*p.z )) ;   // grad(level-eqn) 
            isect.x = n.x ; 
            isect.y = n.y ; 
            isect.z = n.z ;      
        }
        else
        {
            isect.x = zero ; 
            isect.y = zero ; 
            isect.z = t_cand == t1cap ? -one : one ;  
        }
    }
}


