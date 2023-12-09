#pragma once

LEAF_FUNC
float distance_leaf_plane( const float3& pos, const quad& q0 )
{
    const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;   // plane normal direction  
    const float d = q0.f.w ;                                 // distance to origin 
    float pn = dot(pos, n ); 
    float sd = pn - d ;  
    return sd ; 
}


/**
intersect_leaf_plane
-----------------------

* https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection

Equation for points p that are in the plane::

   (p - p0).n = 0      

   p0   : point in plane which is pointed to by normal n vector from origin,  
   p-p0 : vector that lies within the plane, and hence is perpendicular to the normal direction 
   p0.n : d, distance from plane to origin 


   p = o + t v   : parametric ray equation  

   (o + t v - p0).n = 0 

   (p0-o).n  = t v.n

            (p0 - o).n        d - o.n
       t  = -----------  =   -----------
               v.n              v.n  


Special case example : 

* for rays within XZ plane what is the z-coordinate at which rays cross the x=0 "line" ?


                : 
                :    Z
                :    |
                p0   +--X
               /:
              / :
             /  :
            /   :      
           /    :
          +     :
         o     x=0
    

         plane normal  [-1, 0, 0]

    t0 = -o.x/v.x
    z0 =  o.z + t0*v.z 

**/

LEAF_FUNC
bool intersect_leaf_plane( float4& isect, const quad& q0, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;   // plane normal direction  
   const float d = q0.f.w ;                                 // distance to origin 

   float idn = 1.f/dot(ray_direction, n );
   float on = dot(ray_origin, n ); 

   float t_cand = (d - on)*idn ;

   bool valid_intersect = t_cand > t_min ;
   if( valid_intersect ) 
   {
       isect.x = n.x ;
       isect.y = n.y ;
       isect.z = n.z ;
       isect.w = t_cand ; 
   }
   return valid_intersect ; 
}



