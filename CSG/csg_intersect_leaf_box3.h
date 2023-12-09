#pragma once

/**
distance_leaf_box3
--------------------

https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

https://www.youtube.com/watch?v=62-pRVZuS5c

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

**/

LEAF_FUNC
float distance_leaf_box3(const float3& pos, const quad& q0 )
{
    float3 q = make_float3( fabs(pos.x) - q0.f.x/2.f, fabs(pos.y) - q0.f.y/2.f , fabs(pos.z) - q0.f.z/2.f ) ;    
    float3 z = make_float3( 0.f ); 
    float sd = length(fmaxf(q, z)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f ) ;   

#ifdef DEBUG
    printf("//distance_leaf_box3 sd %10.4f \n", sd ); 
#endif
    return sd ; 
}


/**
intersect_leaf_box3
-----------------------

"Fast, Branchless Ray/Bounding Box Intersections"

* https://tavianator.com/2011/ray_box.html

..

    The fastest method for performing ray/AABB intersections is the slab method.
    The idea is to treat the box as the space inside of three pairs of parallel
    planes. The ray is clipped by each pair of parallel planes, and if any portion
    of the ray remains, it intersected the box.


* https://tavianator.com/2015/ray_box_nan.html


Just because the ray intersects the box doesnt 
mean its a usable intersect, there are 3 possibilities::

              t_near       t_far   

                |           |
      -----1----|----2------|------3---------->
                |           |

**/

LEAF_FUNC
bool intersect_leaf_box3(float4& isect, const quad& q0, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
   const float3 bmin = make_float3(-q0.f.x/2.f, -q0.f.y/2.f, -q0.f.z/2.f );   // fullside 
   const float3 bmax = make_float3( q0.f.x/2.f,  q0.f.y/2.f,  q0.f.z/2.f ); 
   const float3 bcen = make_float3( 0.f, 0.f, 0.f ) ;    

#ifdef DEBUG_BOX3
    printf("//intersect_leaf_box3  bmin (%10.4f,%10.4f,%10.4f) bmax (%10.4f,%10.4f,%10.4f)  \n", bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z );  
#endif

   float3 idir = make_float3(1.f)/ray_direction ; 

   // the below t-parameter float3 are intersects with the x, y and z planes of
   // the three axis slab planes through the box bmin and bmax  

   float3 t0 = (bmin - ray_origin)*idir;      //  intersects with bmin x,y,z slab planes
   float3 t1 = (bmax - ray_origin)*idir;      //  intersects with bmax x,y,z slab planes 

   float3 near = fminf(t0, t1);               //  bmin or bmax intersects closest to origin  
   float3 far  = fmaxf(t0, t1);               //  bmin or bmax intersects farthest from origin 

   float t_near = fmaxf( near );              //  furthest near intersect              
   float t_far  = fminf( far );               //  closest far intersect 

   bool along_x = ray_direction.x != 0.f && ray_direction.y == 0.f && ray_direction.z == 0.f ;
   bool along_y = ray_direction.x == 0.f && ray_direction.y != 0.f && ray_direction.z == 0.f ;
   bool along_z = ray_direction.x == 0.f && ray_direction.y == 0.f && ray_direction.z != 0.f ;

   bool in_x = ray_origin.x > bmin.x && ray_origin.x < bmax.x  ;
   bool in_y = ray_origin.y > bmin.y && ray_origin.y < bmax.y  ;
   bool in_z = ray_origin.z > bmin.z && ray_origin.z < bmax.z  ;


   bool has_intersect ;
   if(     along_x) has_intersect = in_y && in_z ;
   else if(along_y) has_intersect = in_x && in_z ; 
   else if(along_z) has_intersect = in_x && in_y ; 
   else             has_intersect = ( t_far > t_near && t_far > 0.f ) ;  // segment of ray intersects box, at least one is ahead


#ifdef DEBUG_BOX3
    printf("//intersect_leaf_box3  along_xyz (%d,%d,%d) in_xyz (%d,%d,%d)   has_intersect %d  \n", along_x, along_y, along_z, in_x, in_y, in_z, has_intersect  );  
    //printf("//intersect_leaf_box3 t_min %10.4f t_near %10.4f t_far %10.4f \n", t_min, t_near, t_far ); 
#endif


   bool has_valid_intersect = false ; 
   if( has_intersect ) 
   {
       float t_cand = t_min < t_near ?  t_near : ( t_min < t_far ? t_far : t_min ) ; 
#ifdef DEBUG_BOX3
       printf("//intersect_leaf_box3 t_min %10.4f t_near %10.4f t_far %10.4f t_cand %10.4f \n", t_min, t_near, t_far, t_cand ); 
#endif

       float3 p = ray_origin + t_cand*ray_direction - bcen ; 

       float3 pa = make_float3(fabs(p.x)/(bmax.x - bmin.x), 
                               fabs(p.y)/(bmax.y - bmin.y), 
                               fabs(p.z)/(bmax.z - bmin.z)) ;

       // discern which face is intersected from the largest absolute coordinate 
       // hmm this implicitly assumes a "box" of equal sides, not a "box3"
       // nope, no problem as the above pa already scales by the fullside so effectivey get a symmetric box 
       // about the origin for the purpose of the comparison
       //
       //
       // Think about intersects onto the unit cube
       // clearly the coordinate with the largest absolute value
       // identifies the x,y or z pair of axes and then 
       // the sign of that gives which face and the outwards normal.
       // Hmm : what about the corner case ?

       float3 n = make_float3(0.f) ;
       if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
       else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
       else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

       if(t_cand > t_min)
       {
           has_valid_intersect = true ; 

           isect.x = n.x ;
           isect.y = n.y ;
           isect.z = n.z ;
           isect.w = t_cand ; 
       }
   }

#ifdef DEBUG_BOX3
   printf("//intersect_leaf_box3 has_valid_intersect %d  isect ( %10.4f %10.4f %10.4f %10.4f)  \n", has_valid_intersect, isect.x, isect.y, isect.z, isect.w ); 
#endif
   return has_valid_intersect ; 
}


