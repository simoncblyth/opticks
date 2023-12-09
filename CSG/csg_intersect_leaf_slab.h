#pragma once

LEAF_FUNC
float distance_leaf_slab( const float3& pos, const quad& q0, const quad& q1 )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
   const float a = q1.f.x ; 
   const float b = q1.f.y ; 
   float pn = dot(pos, n ); 

   //float sd = fmaxf( pn - b, a - pn ) ; 
   float sd = fmaxf( pn - b, pn - a ) ;   // uncertain here

   return sd ; 
}


/**
intersect_leaf_slab
---------------------

One normal, two distances

**/


LEAF_FUNC
bool intersect_leaf_slab( float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   const float a = q1.f.x ; 
   const float b = q1.f.y ; 

   float idn = 1.f/dot(ray_direction, n );
   float on = dot(ray_origin, n ); 

   float ta = (a - on)*idn ;
   float tb = (b - on)*idn ;
   
   float t_near = fminf(ta,tb);  // order the intersects 
   float t_far  = fmaxf(ta,tb);

   float t_cand = t_near > t_min  ?  t_near : ( t_far > t_min ? t_far : t_min ) ; 

   bool valid_intersect = t_cand > t_min ;
   bool b_hit = t_cand == tb ;

   if( valid_intersect ) 
   {
       isect.x = b_hit ? n.x : -n.x ;
       isect.y = b_hit ? n.y : -n.y ;
       isect.z = b_hit ? n.z : -n.z ;
       isect.w = t_cand ; 
   }
   return valid_intersect ; 
}


