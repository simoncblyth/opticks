#pragma once

/**
distance_leaf_phicut
-----------------------

::

      . . . . . . phi0=0.5
      . . . . . . |
      . . . . . . |
      . . . . . . |--> [sinPhi0,-cosPhi0] = [1,0]
      . . . . . . |
      . . . . . . |
      . . . . . . |
      . . . . . . | sd0   (x,y)
      . . . . . . +. . . +
      . . . . . . |      :
      . . . . . . |      sd1
      . . . . . . |      :           [ -sinPhi1, cosPhi1] = [0, 1]
      . . . . . . |      :          ^
      . . . . . . |      :          | 
      . . . . . . +------+----------+--- phi1=2
      . . . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . . . .

**/

LEAF_FUNC
float distance_leaf_phicut( const float3& pos, const quad& q0 )
{
    const float& cosPhi0 = q0.f.x ; 
    const float& sinPhi0 = q0.f.y ; 
    const float& cosPhi1 = q0.f.z ; 
    const float& sinPhi1 = q0.f.w ; 

    // dot products with normal0  [  sinPhi0, -cosPhi0, 0.f ]
    float sd0 = sinPhi0*pos.x - cosPhi0*pos.y ;  

    // dot products with normal1  [ -sinPhi1,  cosPhi1, 0.f ]
    float sd1 = -sinPhi1*pos.x + cosPhi1*pos.y ;  

    return fminf( sd0, sd1 ); 
}


/**
intersect_leaf_phicut
------------------------

Unbounded shape that cuts phi via two half-planes "attached" to the z-axis.
See phicut.py for development and testing of intersect_node_phicut
The phicut planes go through the origin so the equation of the plane is::

    p.n = 0

Intersecting a ray with the plane::

    p = o + t v       parametric ray equation  

   (o + t v).n = 0 

      o.n  = - t v.n

              -o.n
       t  = -----------  
               v.n             

Outwards normals of the two planes::

    n0   (  sin(phi0), -cos(phi0), 0 )   =  ( 0, -1,  0)
    n1   ( -sin(phi1),  cos(phi1), 0 )   =  ( -1, 0,  0)   
 

                phi1=0.5
                  Y
                  | . . . . . . .
                  | . . . . . . .
                  | . . . . . . .
              n1--+ . . . . . . .
                  | . . . . . . .
                  | . . . . . . . 
   phi=1.0 . . . .O----+-------- X    phi0=0 
                  .    |
                  .    n0
                  .
                  .
                  .
                phi=1.5


Normal incidence is no problem::

          Y
          |
          | 
          | 
          |     (10,0,0)
          O------+---------- X
                 |      
                 |
                 |
                 * 
                 normal        ( 0,  -1, 0)
                 ray_origin    (10, -10, 0)  .normal =  10   
                 ray_direction ( 0,   1, 0)  .normal  = -1  

                 t = -10/(-1) = 10

Disqualify rays with direction within the plane::
  
       normal        (  0, -1, 0)      
       ray_origin    ( 20, 0,  0)   .normal     0
       ray_direction ( -1, 0,  0)   .normal     0


Edge vectors P, Q and radial direction vector R::

   P  ( cosPhi0, sinPhi0, 0 )        phi0=0   (1, 0, 0)  phi0=1  (-1, 0, 0)  
   Q  ( cosPhi1, sinPhi1, 0 )  
   R  (  d.x   ,   d.y  , 0 )        d.z is ignored, only intersected in radial direction within XY plane

Use conventional cross product sign convention  A ^ B =  (Ax By - Bx Ay ) k the products are::
 
   PQ = P ^ Q = cosPhi0*sinPhi1 - cosPhi1*sinPhi0 
   PR = P ^ R = cosPhi0*d.y - d.x*sinPhi0 
   QR = Q ^ R = cosPhi1*d.y - d.x*sinPhi1 

Note that as R is not normalized the PR and QR cross products are only proportional to the sine of the angles.
See CSG/tests/cross2D_angle_range_without_trig.py for exploration of 2D cross product 

Alternative way of looking at this is with the sine of angle subtraction identity::

    sin(a-b) = sin(a)cos(b)-cos(a)sin(b)  

When rays have non-zero x,y direction components the 2D cross products 
between the direction vectors and the phi edge vectors determines
if the rays will exit the unbounded shape at infinity.

When rays are purely in Z direction this approach does not work as PR and QR 
will always be zero because d.x and d.y are zero.  
So in this zpure case the (o.x, o.y) are used instead of (d.x, d.y)
Can think of vectors from origin to the (o.x, o.y) ray origin which can 
serve the same purpose of determining whether the ray will exit the 
shape at infinity or not. 

                                                        NO 
            . . . . . . . . . . . .|                   /
            . . . . . . . . . . . .|                  /
            . . . . . . . . . . . .|                 /
            . . . . . . . . . . . .|                /
            . . . . . . . . . . phi0=0.5           /
            . . . . . . . . . . . .|              /
            . . . . . . . . . . . .|             0  sd > 0.f     
            . . . . . . . . . . . .|             
            . . . . . 0 . . . . . .|             
            . . . . ./. . . . . . .|             
            . . . . / . . . . . . .+------------ phi1=2.0 ---  X   
            . . . ./. . . . . . . . . . . . . . . . . . . . . . .
            . . . / . . . . . . . . . . . . . . . . . . . . . . .
            . . ./. . . . . . . . . . . . . . . . . . . . . . . .
            . . / . . . . . . . . . . . . . . . .0------------------ YES UNBOUNDED_EXIT AT INFINITY
            . ./. . . . . . . . . . . . . . . .sd < 0.f . . . . .
            . / . . . . . . . . . . . . . . . . . . . . . . . . .
            ./. . . . . . . . . . . . . . . . . . . . . . . . . .
            / . . . . . . . . . . . . . . . . . . . . . . . . . .
           / 
         YES



Clockwise rotation by phi (flipping sign):: 

  | x' |    |  cosPhi   sinPhi |   | x  |
  |    | =  |                  |   |    |
  | y' |    | -sinPhi   cosPhi |   | y  | 
 

    x' = cosPhi x + sinPhi y 

    y' = -sinPhi x + cosPhi y 


Within edgezone want to avoid two separate decisions as to which face gets hit 
otherwise numerical imprecision will result in inconsistent decisions causing misses (white seams) along the edge. 



Thoughts on implementing phicut by CSG combination of two halfspaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

phi1 - phi0 > 1.0 
    A+B      : union of two halfspaces   
    !(!A.!B) : complement of intersection of the two halfspaces complemented 

::
      
        
            A | !A
          phi0=0.5
       . . . .|
       . . . .|               
       . . . .|         
       . . . .|          
       . . . .|            !B
       ~~~~~~~+--------- phi1=2.0
       . . . .:. . . . .    B
       . . . .:. . . . .
       . . . .:. . . . .
       . . . .:. . . . .

phi1 - phi0 < 1.0 : 

    A.B      :  intersection of two halfspaces 
    !(!A+!B) : complement of union of two halfspaces complemented

           !A | A
           phi1=0.5
              | . . . . .         
              | . . . . .         
              | . . . . .        
              | . . . . .  B         
       ~~~~~~~+--------- phi0=0.0
              :            !B 
              :
              :
              :
**/


LEAF_FUNC
bool intersect_leaf_phicut( float4& isect, const quad& q0, const float t_min, const float3& o, const float3& d )
{
    const float& cosPhi0 = q0.f.x ; 
    const float& sinPhi0 = q0.f.y ; 
    const float& cosPhi1 = q0.f.z ; 
    const float& sinPhi1 = q0.f.w ; 

#ifdef DEBUG
    printf("//intersect_leaf_phicut  q0.f (%10.4f %10.4f   %10.4f %10.4f )   cosPhi0 sinPhi0   cosPhi1 sinPhi1      t_min  %10.4f \n", cosPhi0, sinPhi0, cosPhi1, sinPhi1, t_min ); 
#endif

    const float sd0 =  sinPhi0*(o.x+t_min*d.x) - cosPhi0*(o.y+t_min*d.y) ;  
    const float sd1 = -sinPhi1*(o.x+t_min*d.x) + cosPhi1*(o.y+t_min*d.y) ;  
    float sd = fminf( sd0, sd1 );   // signed distance at t_min, sd < 0.f means are within the phicut shape : ie between the planes

#ifdef DEBUG
    printf("//intersect_leaf_phicut  sd0 %10.4f sd1 %10.4f sd  %10.4f \n", sd0, sd1, sd ); 
#endif

    const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    const bool zpure = d.x == 0.f && d.y == 0.f ;   
    const float PR = cosPhi0*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    const float QR = cosPhi1*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi1  ;  
    const bool unbounded_exit = sd < 0.f &&  ( PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f ))  ;   

#ifdef DEBUG
    printf("//intersect_leaf_phicut  PQ %10.4f zpure %d \n", PQ, zpure ); 
    printf("//intersect_leaf_phicut  PR %10.4f  QR %10.4f unbounded_exit %d \n", PR,QR, unbounded_exit ); 
#endif
 
    // dotproduct with outward normal [ sinPhi0, -cosPhi0, 0 ]  of phi0 plane    eg [1,0,0] for pacmanpp
    const float t0 = -( o.x*sinPhi0 + o.y*(-cosPhi0) )/ ( d.x*sinPhi0 + d.y*(-cosPhi0) ) ;  // -o.x/d.x 
    const float x0 = o.x+t0*d.x ; 
    const float y0 = o.y+t0*d.y ; 
    const float s0x =  cosPhi0*x0 + sinPhi0*y0 ; 

#ifdef DEBUG
    const float s0y = -sinPhi0*x0 + cosPhi0*y0 ; 
    printf("//intersect_leaf_phicut  t0 %10.4f  (x0,y0) (%10.4f, %10.4f)   (s0x,s0y) (%10.4g, %10.4g) \n", t0, x0,y0, s0x,s0y ); 
#endif

    // dotproduct with outward normal [ -sinPhi1, cosPhi1, 0 ]  of phi1 plane    eg [0,1,0] for pacmanpp
    const float t1 = -( o.x*(-sinPhi1) + o.y*cosPhi1 )/( d.x*(-sinPhi1) + d.y*cosPhi1 ) ;  
    const float x1 = o.x+t1*d.x ;                  
    const float y1 = o.y+t1*d.y ;                  
    const float s1x =  cosPhi1*x1 + sinPhi1*y1 ; 

#ifdef DEBUG
    const float s1y = -sinPhi1*x1 + cosPhi1*y1 ; 
    printf("//intersect_leaf_phicut  t1 %10.4f  (x1,y1) (%10.4f, %10.4f)   (s1x,s1y) (%10.4f, %10.4f) \n", t1, x1,y1, s1x,s1y ); 
#endif

    const float epsilon = 1e-4f ; 
    bool safezone = ( fabsf(s0x) > epsilon && fabsf(s1x) > epsilon ) ;
    const float t0c = ( s0x >= 0.f && t0 > t_min) ? t0 : RT_DEFAULT_MAX ; 
    const float t1c = ( s1x >= 0.f && t1 > t_min) ? t1 : RT_DEFAULT_MAX ; 
    const float t_cand = safezone ? fminf( t0c, t1c ) : ( s0x >= 0.f ? t0c : t1c ) ;
    const bool valid_intersect = t_cand > t_min && t_cand <  RT_DEFAULT_MAX ; 

    /*
       0. s0x s1x are -phi0 and -phi1 rotated xprime coordinates of intersect positions
          it might be tempting to simply match signs of intersect x or y 
          but that does not work in general for any phi0 phi1 angle  

          * eg pacmanpp +x   x1:-inf y1:nan s1x:nan s1y:nan

       1. s0x s1x can become nan (from t0 or t1 inf or -inf and inf*0.f->nan or -inf*0.f->nan ) 
          and comparisons with nan always give false. So arrange direction of the comparisons 
          such that true yields the root in order that a false from nan comparison does what 
          is needed and disqualifies the root. 
    
       2. t0, t1 can be -inf/inf/nan so must check them against t_min individually 

          * when t0 OR t1 is inf the s0x OR s1x will be nan 

       3. must prevent nan from one root infecting the other, by quashing it before root comparison

       4. close to knife edge numerical imprecision makes inconsistencies between sign comparisons on s0x and s1x likely 
          (such inconsistencies cause seams lines of MISS-es along the knife edge between the phi planes)

          * having unexpected holes in geometry is more of a problem than having normals at edge swap between planes
          * hence : prevent inconsistent decisions by making just one decision over which plane is hit when at knife edge

          * TODO: current approach makes the knife edge decision based on s0x (an arbitrary choice)
            it would be better to choose based on the s0x or s1x with the larger absolute value, 
            but need to fish for which is which to pick t0c or t1c::

                 fmaxf(abs(s0x), abs(s1x)) == abs(s0x) ? t0c : t1c 

    */ 


#ifdef DEBUG
    printf("//intersect_leaf_phicut t0c %10.4f t1c %10.4f safezone %d t_cand %10.4f valid_intersect %d  unbounded_exit %d \n", t0c, t1c, safezone, t_cand, valid_intersect, unbounded_exit );  
#endif

    if( valid_intersect ) 
    {
        isect.x = t_cand == t1 ? -sinPhi1 :  sinPhi0 ; 
        isect.y = t_cand == t1 ?  cosPhi1 : -cosPhi0 ;  
        isect.z = 0.f ; 
        isect.w = t_cand ; 
    }
    else if( unbounded_exit )
    {
        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity 
    }
    return valid_intersect ; 
}









LEAF_FUNC
bool intersect_leaf_phicut_dev( float4& isect, const quad& q0, const float t_min, const float3& o, const float3& d )
{
    const float& cosPhi0 = q0.f.x ; 
    const float& sinPhi0 = q0.f.y ; 
    const float& cosPhi1 = q0.f.z ; 
    const float& sinPhi1 = q0.f.w ; 

    const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    bool zpure = d.x == 0.f && d.y == 0.f ;   
    const float PR = cosPhi0*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    const float QR = cosPhi1*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi1  ;  
    bool unbounded_exit = PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f )  ;   

#ifdef DEBUG
    printf("//intersect_leaf_phicut q0.f  (%10.4f %10.4f %10.4f %10.4f) %s t_min %10.4f \n" , q0.f.x, q0.f.y, q0.f.z, q0.f.w, "cosPhi0/sinPhi0/cosPhi1/sinPhi1", t_min  ) ; 
    printf("//intersect_leaf_phicut d.xyz ( %10.4f %10.4f %10.4f ) zpure %d \n", d.x, d.y, d.z, zpure  ); 
    printf("//intersect_leaf_phicut PQ %10.4f cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve:angle less than pi, -ve:angle greater than pi \n", PQ );
    printf("//intersect_leaf_phicut PR %10.4f cosPhi0*d.y - d.x*sinPhi0 \n", PR );
    printf("//intersect_leaf_phicut QR %10.4f cosPhi1*d.y - d.x*sinPhi1 \n", QR );
    printf("//intersect_leaf_phicut unbounded_exit %d \n", unbounded_exit ); 
#endif

    // setting t values to t_min disqualifies that intersect
    // dot products with normal0  [ sinPhi0, -cosPhi0, 0.f ]
      
    /*
    // Old careful approach works, but unneeded rotation flops for checking the side, 
    // can just check signbit of x intersect matches the sigbit of the cosPhi

    float d_n0 = d.x*sinPhi0 + d.y*(-cosPhi0) ; 
    float o_n0 = o.x*sinPhi0 + o.y*(-cosPhi0) ; 
    float t0 = d_n0 == 0.f ? t_min : -o_n0/d_n0 ;                 // perhaps could avoid the check if side0 became -inf ? 

    float side0 = o.x*cosPhi0 + o.y*sinPhi0 + ( d.x*cosPhi0 + d.y*sinPhi0 )*t0 ;  
    if(side0 < 0.f) t0 = t_min ; 

    float d_n1 = d.x*(-sinPhi1) + d.y*cosPhi1 ; 
    float o_n1 = o.x*(-sinPhi1) + o.y*cosPhi1 ; 
    float t1 = d_n1 == 0.f ? t_min : -o_n1/d_n1 ; 

    float side1 = o.x*cosPhi1 + o.y*sinPhi1 + ( d.x*cosPhi1 + d.y*sinPhi1 )*t1 ;  
    if(side1 < 0.f) t1 = t_min ; 
    */


    /*

    // Medium careful approach works, accepting t0 and t1 becoming infinity for some ray directions
    // but still carefully ordering the roots. 
    // 
    float t0 = -(o.x*sinPhi0 + o.y*(-cosPhi0))/ ( d.x*sinPhi0 + d.y*(-cosPhi0) ) ; 
    if(signbit(o.x+t0*d.x) != signbit(cosPhi0)) t0 = t_min ; 

    float t1 = -(o.x*(-sinPhi1) + o.y*cosPhi1)/(d.x*(-sinPhi1) + d.y*cosPhi1 ) ; 
    if(signbit(o.x+t1*d.x) != signbit(cosPhi1)) t1 = t_min ; 

    float t_near = fminf(t0,t1);  // order the intersects 
    float t_far  = fmaxf(t0,t1);
    float t_cand = t_near > t_min  ?  t_near : ( t_far > t_min ? t_far : t_min ) ; 
    bool valid_intersect = t_cand > t_min ;

   */ 

    
    //  Lucas wild abandon approach gives bad intersects 
    //  Select isect in the bad region::
    //
    //      SPHI=0.24,1.76 ./csg_geochain.sh ana
    //      SPHI=0.24,1.76 IXYZ=4,4,0 ./csg_geochain.sh ana
    //
    //   Q: why ? its little different to the above which has no problem 
    //   A: bug t_cand < t_min which should be t_cand <= t_min   
    //
    //   https://tavianator.com/2011/ray_box.html
    //    0*inf = nan
    //

    
    float t_cand = -( o.x*sinPhi0 + o.y*(-cosPhi0) )/ ( d.x*sinPhi0 + d.y*(-cosPhi0) ) ; 
    // o on phi0 line => t_cand.0  -0.f 
    // o on phi0 line, d along line => t_cand.0  0/0 = nan 

#ifdef DEBUG
    //printf("//intersect_leaf_phicut ( o.x*sinPhi0 + o.y*(-cosPhi0)    %10.4f \n", ( o.x*sinPhi0 + o.y*(-cosPhi0)) ); 
    //printf("//intersect_leaf_phicut ( d.x*sinPhi0 + d.y*(-cosPhi0)    %10.4f \n", ( d.x*sinPhi0 + d.y*(-cosPhi0)) ); 
    printf("//intersect_leaf_phicut t_cand.0 %10.4f t_min %10.4f \n", t_cand, t_min  ); 
    //printf("//intersect_leaf_phicut o.x+t_cand*d.x  %10.4f \n", o.x+t_cand*d.x ); 
    //printf("//intersect_leaf_phicut signbit(o.x+t_cand*d.x)  %d \n", signbit(o.x+t_cand*d.x) ); 
    //printf("//intersect_leaf_phicut signbit(o.x+t_cand*d.x) != signbit(cosPhi0)  %d \n", signbit(o.x+t_cand*d.x) != signbit(cosPhi0) ); 
    //printf("//intersect_leaf_phicut t_cand.0 < t_min : %d \n", t_cand < t_min ); 
    //printf("//intersect_leaf_phicut t_cand.0 > t_min : %d \n", t_cand > t_min ); 
    //printf("//intersect_leaf_phicut t_cand.0 == t_min : %d \n", t_cand == t_min ); 
    //printf("//intersect_leaf_phicut signbit(o.x+t_cand*d.x) != signbit(cosPhi0) || t_cand.0 < t_min : %d \n",  signbit(o.x+t_cand*d.x) != signbit(cosPhi0) || t_cand < t_min ); 

    {
        /*
        in XZ projection testing deciding based on ipos_x yields spurious due to ipos_x going slightly wrong sign 
        switching to ipos_y avoids the problem : WHY ?
        */
        float ipos_x = (o.x+t_cand*d.x) ; 
        float ipos_y = (o.y+t_cand*d.y) ; 
        float ipos_z = (o.z+t_cand*d.z) ; 
        bool x_wrong_side = ipos_x*cosPhi0 < 0.f ; 
        bool y_wrong_side = ipos_y*sinPhi0 < 0.f ; 
        bool too_close =  t_cand <= t_min ; 
        printf("//intersect_leaf_phicut ipos_x %10.4f ipos_x*1e6f %10.4f  cosPhi0 %10.4f  x_wrong_side %d \n", ipos_x, ipos_x*1e6f, cosPhi0, x_wrong_side ); 
        printf("//intersect_leaf_phicut ipos_y %10.4f ipos_y*1e6f %10.4f  sinPhi0 %10.4f  y_wrong_side %d \n", ipos_y, ipos_y*1e6f, sinPhi0, y_wrong_side ); 
        printf("//intersect_leaf_phicut ipos_z %10.4f ipos_z*1e6f %10.4f                                  \n", ipos_z, ipos_z*1e6f ); 
        printf("//intersect_leaf_phicut t_cand %10.4f t_min %10.4f too_close %d \n", t_cand, t_min, too_close ); 
    }


/*
rays along the phi0 line get t_cand nan from 0/0 
BUT as all comparisons with nan yield false the below 
also note that signbit(nan) == true which would also invalidate +ve cosPhi0 
*/
#endif

    if((o.y+t_cand*d.y)*sinPhi0 < 0.f || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ; 

    //if((o.x+t_cand*d.x+1e-4)*cosPhi0 < 0.f || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ; 
    //if((o.x+t_cand*d.x)*cosPhi0 < 0.f || t_cand < t_min ) t_cand = RT_DEFAULT_MAX ;         // t_cand < t_min YIELDS SPURIOUS INTERSECTS 
/*
Disqualify wrong side or too close  
                                        ^^^^^^^^^^^^^^^^^^  

1. must be  t_cand <= t_min to avoid spurious intersects in CSG combination 
   for rays starting on phi0 line which have t_cand -0.f

   * this is because NOT("t_cand > t_min") -> "t_cand <= t_min"  

2. At first glance using signbit seems better, but signbit(nan) == true 
   which means that the t_cand nan does not propagate to the comparison 
   so the invalidation will only happen for +ve cosPhi0 

   signbit(o.x+t_cand*d.x) != signbit(cosPhi0) 

   * actually not invalidating here doesnt matter as the t_cand nan will 
     simply end up giving a false for valid_intersect 

   * product (o.x+t_cand*d.x)*cosPhi0  will be nan for t_cand nan and 
     all comparisons with nan give false so t_cand will stay nan here

   * all comparisons with nan always returns false 
     but that does not kill the entire short circuit OR 
     (see sysrap/tests/signbitTest.cc)

3. at first glance may think could make the disqualification based only 
   x or y of intersect BUT that will nor work for all phi0 phi1 angles.
   So in general it is necessary to rotate the intersect (x,y) positions 
   by -phi0 and -phi1 to place it back on the x line. 



 


*/

#ifdef DEBUG
    printf("//intersect_leaf_phicut t_cand.1 %10.4f \n", t_cand ); 
#endif

    const float t1 = -( o.x*(-sinPhi1) + o.y*cosPhi1 )/( d.x*(-sinPhi1) + d.y*cosPhi1 ) ; 
    if((o.x + t1*d.x)*cosPhi1 > 0.f && t1 > t_min ) t_cand = fminf( t1, t_cand );  
    bool valid_intersect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;  

#ifdef DEBUG
    //printf("//intersect_leaf_phicut t1_num  ( o.x*(-sinPhi1) + o.y*cosPhi1 )  : %10.4f \n", ( o.x*(-sinPhi1) + o.y*cosPhi1 ) ); 
    //printf("//intersect_leaf_phicut t1_den  ( d.x*(-sinPhi1) + d.y*cosPhi1 )  : %10.4f \n", ( d.x*(-sinPhi1) + d.y*cosPhi1 ) );  
    //printf("//intersect_leaf_phicut t1      %10.4f \n", t1 ); 
    //printf("//intersect_leaf_phicut  signbit(o.x + t1*d.x) == signbit(cosPhi1) && t1 > t_min  : %d \n", signbit(o.x + t1*d.x) == signbit(cosPhi1) && t1 > t_min ); 
    printf("//intersect_leaf_phicut t_cand.2 %10.4f valid_intersect %d \n", t_cand, valid_intersect  ); 
#endif

    if( valid_intersect ) 
    {
        isect.x = t_cand == t1 ? -sinPhi1 :  sinPhi0 ; 
        isect.y = t_cand == t1 ?  cosPhi1 : -cosPhi0 ;  
        isect.z = 0.f ; 
        isect.w = t_cand ; 
    }
    else if( unbounded_exit )
    {
        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity 
    }
    return valid_intersect ; 
}




/**
intersect_leaf_phicut_lucas
----------------------------

Simon's comments on intersect_leaf_phicut_lucas

1. code must stay readable

2. scrunching up code to make it shorter, does not make it faster.
   The only way to tell if its faster is to measure it in the context 
   in which you want to use it.
   
3. code you write is not what gets run : the compiler performs lots of optimizations on it first 

   * you should of course help the compiler to know what your intentions are 
     using const and references to avoid needless copies while still retaining readability   

4. for rays starting inside the planes your t_cand and t1 will get divisions by zero and infinities

   * that is not necessarily a problem : but you have to think thru what will happen and test the cases 

5. disqualification of candidates needs to use t_cand > t_min as for this shape to work 
   in CSG combination it must perform as expected as t_min is varied 

   * using the below commandline with a test geometry using phicut will 
     produce renders that you can use to see how it performs as t_min is varied:: 

         EYE=1,1,1 TMIN=0.5 ./cxr_geochain.sh 

   * the required behaviour is for the t_min to cut away the solid in an expanding sphere 
     shape when using perspective projection (plane when using parallel projection)

 
            |        /
            |       /
            |      +
            |     /:
            |    / :
            |   /  :
            | [1]--:- - - - - 0
            | /:   :    accepted sign(xi) == sign(cosPhi1)  
            |/ :   :
   . . . . .O--:-- +-----+  X
           .:  :xi :cosPhi1
          . :
        *1*- - - - - - - - - 0     disqualified "other half" intersect 
        .   :      (-ve x)*(+ve cosPhi1 ) => -ve => disqualified 
       .    :          
      .     :
     .      :

**/

LEAF_FUNC
bool intersect_leaf_phicut_lucas(float4& isect, const quad& angles, const float& t_min, const float3& ray_origin, const float3& ray_direction)
{   //for future reference: angles.f has: .x = cos0, .y = sin0, .z = cos1, .w = sin1

    float t_cand = -(angles.f.y * ray_origin.x - angles.f.x * ray_origin.y) / (angles.f.y * ray_direction.x - angles.f.x * ray_direction.y);
    //using t_cand saves unnecessary definition (t_cand = -( Norm0 * Or ) / ( Norm0 * Dir ), ratio of magnitudes in the direction of plane)

    if (t_cand * angles.f.x * ray_direction.x  + angles.f.x * ray_origin.x < 0.f || t_cand <= t_min)
        t_cand = RT_DEFAULT_MAX; // disqualify t_cand for wrong side 

    const float t1 = -(-angles.f.w * ray_origin.x + angles.f.z * ray_origin.y) / (-angles.f.w * ray_direction.x + angles.f.z * ray_direction.y);
    if (t1 * angles.f.z * ray_direction.x + angles.f.z * ray_origin.x > 0.f && t1 > t_min)
        t_cand = fminf(t1, t_cand);

     //   angles.f.z * ( t1 * ray_direction.x + ray_origin.x )
     //   cosPhi1 * (intersection_x) > 0  


    const bool valid = t_cand < RT_DEFAULT_MAX;
    if (valid)
    {
        isect.x = t_cand == t1 ? -angles.f.w: angles.f.y;
        isect.y = t_cand == t1 ? angles.f.z : -angles.f.x;
        isect.z = 0.f;
        isect.w = t_cand;
    }

    return valid;
}



