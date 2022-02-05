#pragma once

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
 

                phi=0.5
                  Y
                  
                  |      
                  |
              n1--+
                  |
                  |
   phi=1.0 . . . .O----+----- X    phi=0, 2 
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


**/

LEAF_FUNC
bool intersect_leaf_phicut( float4& isect, const quad& q0, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float& cosPhi0 = q0.f.x ; 
    const float& sinPhi0 = q0.f.y ; 
    const float& cosPhi1 = q0.f.z ; 
    const float& sinPhi1 = q0.f.w ; 

    // setting t values to t_min disqualifies that intersect
    // dot products with normal0  [ sinPhi0, -cosPhi0, 0.f ]
    float d_n0 = ray_direction.x*sinPhi0 + ray_direction.y*(-cosPhi0) ; 
    float o_n0 = ray_origin.x*sinPhi0 + ray_origin.y*(-cosPhi0) ; 
    float t0 = d_n0 == 0.f ? t_min : -o_n0/d_n0 ;                 // perhaps could avoid the check if side0 became -inf ? 
    float side0 = ray_origin.x*cosPhi0 + ray_origin.y*sinPhi0 + ( ray_direction.x*cosPhi0 + ray_direction.y*sinPhi0 )*t0 ;  
    if(side0 < 0.f) t0 = t_min ; 

    // dot products with normal1   [ -sinPhi1,  cosPhi1, 0.f ]
    float d_n1 = ray_direction.x*(-sinPhi1) + ray_direction.y*cosPhi1 ; 
    float o_n1 = ray_origin.x*(-sinPhi1) + ray_origin.y*cosPhi1 ; 
    float t1 = d_n1 == 0.f ? t_min : -o_n1/d_n1 ; 
    float side1 = ray_origin.x*cosPhi1 + ray_origin.y*sinPhi1 + ( ray_direction.x*cosPhi1 + ray_direction.y*sinPhi1 )*t1 ;  
    if(side1 < 0.f) t1 = t_min ; 

#ifdef DEBUG
    printf("//intersect_leaf_phicut d_n0 %10.3f o_n0 %10.3f t0 %10.3f  side0 %10.3f  \n", d_n0, o_n0, t0, side0 ); 
    printf("//intersect_leaf_phicut d_n1 %10.3f o_n1 %10.3f t1 %10.3f  side1 %10.3f  \n", d_n1, o_n1, t1, side1 ); 
#endif

    float t_near = fminf(t0,t1);  // order the intersects 
    float t_far  = fmaxf(t0,t1);
    float t_cand = t_near > t_min  ?  t_near : ( t_far > t_min ? t_far : t_min ) ; 

    bool valid_intersect = t_cand > t_min ;

    if( valid_intersect ) 
    {
        isect.x = t_cand == t1 ? -sinPhi1 :  sinPhi0 ; 
        isect.y = t_cand == t1 ?  cosPhi1 : -cosPhi0 ;  
        isect.z = 0.f ; 
        isect.w = t_cand ; 
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

