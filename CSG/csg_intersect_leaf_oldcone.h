#pragma once

/**
intersect_leaf_oldcone
========================

Suspect this cone implementation has issues with axial rays 
and rays onto "corners" 

Notice problems for rays along axis line thru apex 
and for rays in -z direction onto the edge between the endcap 
and quadratic sheet marked (*) on below::


                       *
                       |     [0,0,0]
       ----------------A------------------           
                      / \
                     /   \
                    /     \
                 * /       \ * 
                 |/         \|
                 +-----------+         z = z2
                /     r2      \
               /               \
              /                 \
             /                   \
            +---------------------+   z = z1
      [-300,0,-300]    r1       [+300,0,-300]


TODO: investigate and see if some special casing can avoid the issues.



    cone with apex at [0,0,z0]  and   r1/(z1-z0) = tanth  for any r1,z1 on the cone
    
        x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
        x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
    
      Gradient:    [2x, 2y, (-2z tanth^2) + 2z0 tanth^2 ] 



To find ray intersect with infinite cone insert parametric ray
into definining eqn of cone, giving a quadratic in t::

    (o.x+ t d.x)^2 + (o.y + t d.y)^2 - (o.z - z0 + t d.z)^2 tth2 = 0 
     
    c2 t^2 + 2 c1 t + c0 = 0 

**/

LEAF_FUNC
bool intersect_leaf_oldcone( float4& isect, const quad& q0, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float r1 = q0.f.x ; 
    float z1 = q0.f.y ; 
    float r2 = q0.f.z ; 
    float z2 = q0.f.w ;   // z2 > z1

    float tth = (r2-r1)/(z2-z1) ;
    float tth2 = tth*tth ; 
    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex

#ifdef DEBUG_CONE
    printf("//intersect_leaf_oldcone r1 %10.4f z1 %10.4f r2 %10.4f z2 %10.4f : z0 %10.4f \n", r1, z1, r2, z2, z0 );  
#endif
 
    float r1r1 = r1*r1 ; 
    float r2r2 = r2*r2 ; 

    const float3& o = ray_origin ;
    const float3& d = ray_direction ;

    // collecting terms to form coefficients of the quadratic : c2 t^2 + 2 c1 t + c0 = 0 

    float c2 = d.x*d.x + d.y*d.y - d.z*d.z*tth2 ;
    float c1 = o.x*d.x + o.y*d.y - (o.z-z0)*d.z*tth2 ; 
    float c0 = o.x*o.x + o.y*o.y - (o.z-z0)*(o.z-z0)*tth2 ;
    float disc = c1*c1 - c0*c2 ; 

    // when c0 or c2 are small, sdisc will be very close to c1 
    // resulting in catastrophic precision loss for the root that 
    // is obtained by subtraction of two close values
    // => because of this need to use robust_quadratic_roots 
    // as done in csg_intersect_leaf_newcone.h 

#ifdef DEBUG_CONE
    printf("//intersect_leaf_oldcone c2 %10.4f c1 %10.4f c0 %10.4f disc %10.4f disc > 0.f %d : tth %10.4f \n", c2, c1, c0, disc, disc>0.f, tth  );  
#endif
 
    // * cap intersects (including axial ones) will always have potentially out of z-range cone intersects 
    // * cone intersects will have out of r-range plane intersects, other than rays within xy plane

    bool valid_isect = false ;
 
    if(disc > 0.f)  // has intersects with infinite cone
    {
        float sdisc = sqrtf(disc) ;   
        float root1 = (-c1 - sdisc)/c2 ;    
        float root2 = (-c1 + sdisc)/c2 ;  

        float root1p = root1 > t_min ? root1 : RT_DEFAULT_MAX ;  
        float root2p = root2 > t_min ? root2 : RT_DEFAULT_MAX ; 
        // disqualify -ve roots from mirror cone thats behind you immediately 

        // order the roots 
        float t_near = fminf( root1p, root2p );
        float t_far  = fmaxf( root1p, root2p );  


#ifdef DEBUG_CONE
        printf("//intersect_leaf_oldcone c0 %10.4g c1 %10.4g c2 %10.4g t_near %10.4g t_far %10.4g sdisc %10.4f "
                "(-c1-sdisc) %10.4g (-c1+sdisc) %10.4g \n", 
              c0, c1, c2, t_near, t_far, sdisc, (-c1-sdisc), (-c1+sdisc)  );  

        float t_near_alt, t_far_alt, disc_alt, sdisc_alt ;   
        robust_quadratic_roots(t_near_alt, t_far_alt, disc_alt, sdisc_alt, c2, c1, c0 ) ;

        printf("//intersect_leaf_oldcone t_near_alt %10.4g t_far_alt %10.4g t_near_alt-t_near %10.4g t_far_alt-t_far %10.4g \n", 
             t_near_alt, t_far_alt, (t_near_alt-t_near), (t_far_alt-t_far) );
#endif


        float z_near = o.z+t_near*d.z ; 
        float z_far  = o.z+t_far*d.z ; 

        t_near = z_near > z1 && z_near < z2  && t_near > t_min ? t_near : RT_DEFAULT_MAX ; // disqualify out-of-z
        t_far  = z_far  > z1 && z_far  < z2  && t_far  > t_min ? t_far  : RT_DEFAULT_MAX ; 

        // intersects with cap planes
        float idz = 1.f/d.z ; 
        float t_cap1 = d.z == 0.f ? RT_DEFAULT_MAX : (z1 - o.z)*idz ;   // d.z zero means no z-plane intersects
        float t_cap2 = d.z == 0.f ? RT_DEFAULT_MAX : (z2 - o.z)*idz ;

        // radii squared at cap intersects  
        float r_cap1 = (o.x + t_cap1*d.x)*(o.x + t_cap1*d.x) + (o.y + t_cap1*d.y)*(o.y + t_cap1*d.y) ;  
        float r_cap2 = (o.x + t_cap2*d.x)*(o.x + t_cap2*d.x) + (o.y + t_cap2*d.y)*(o.y + t_cap2*d.y) ;  

        t_cap1 = r_cap1 < r1r1 && t_cap1 > t_min ? t_cap1 : RT_DEFAULT_MAX ;  // disqualify out-of-radius
        t_cap2 = r_cap2 < r2r2 && t_cap2 > t_min ? t_cap2 : RT_DEFAULT_MAX ; 
 
        float t_capn = fminf( t_cap1, t_cap2 );    // order caps
        float t_capf = fmaxf( t_cap1, t_cap2 );

        // NB use of RT_DEFAULT_MAX to represent disqualified
        // roots is crucial to picking closest  qualified root with 
        // the simple fminf(tt) 

        float4 tt = make_float4( t_near, t_far, t_capn, t_capf );
        float t_cand = fminf(tt) ; 
        
        valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
        if(valid_isect)
        {
            if( t_cand == t_cap1 || t_cand == t_cap2 )
            {
                isect.x = 0.f ; 
                isect.y = 0.f ;
                isect.z =  t_cand == t_cap2 ? 1.f : -1.f  ;   
            }
            else
            { 
                //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
                //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
                //
                //   Gradient:    [2x, 2y, (-2z + 2z0) tanth^2 ] 
                //   Gradient:    2*[x, y, (z0-z) tanth^2 ] 
                float3 n = normalize(make_float3( o.x+t_cand*d.x, o.y+t_cand*d.y, (z0-(o.z+t_cand*d.z))*tth2  ))  ; 
                // huh : surely simpler way to get normal, just from cone param ?

                isect.x = n.x ; 
                isect.y = n.y ;
                isect.z = n.z ; 
            }
            isect.w = t_cand ; 
        }
    }
    return valid_isect ; 
}


