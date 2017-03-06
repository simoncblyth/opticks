
// https://tavianator.com/fast-branchless-raybounding-box-intersections/
// https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/


static __device__ void intersect_box_branching(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
   const float3 bmin = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
   const float3 bmax = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   float t1, t2 ;
   float tmin = -CUDART_INF_F ; 
   float tmax =  CUDART_INF_F ; 

   //rtPrintf(" ray.origin %f %f %f ray.direction %f %f %f \n ", ray.origin.x,    ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z );
   //rtPrintf(" bmin %f %f %f bmax %f %f %f \n", bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z );

   if(ray.direction.x != 0.f)
   {
       t1 = (bmin.x - ray.origin.x)/ray.direction.x ;
       t2 = (bmax.x - ray.origin.x)/ray.direction.x ;
 
       tmin = max( tmin, min(t1, t2) );
       tmax = min( tmax, max(t1, t2) );

       //rtPrintf(" dir.x %f t1 %f t2 %f tmin %f tmax %f \n", ray.direction.x, t1, t2, tmin, tmax );
   }

   if(ray.direction.y != 0.f)
   {
       t1 = (bmin.y - ray.origin.y)/ray.direction.y ;
       t2 = (bmax.y - ray.origin.y)/ray.direction.y ;
 
       tmin = max( tmin, min(t1, t2) );
       tmax = min( tmax, max(t1, t2) );

       //rtPrintf(" dir.y %f t1 %f t2 %f tmin %f tmax %f \n", ray.direction.y, t1, t2, tmin, tmax );
   }

   if(ray.direction.z != 0.f)
   {
       t1 = (bmin.z - ray.origin.z)/ray.direction.z ;
       t2 = (bmax.z - ray.origin.z)/ray.direction.z ;
 
       tmin = max( tmin, min(t1, t2) );
       tmax = min( tmax, max(t1, t2) );

       //rtPrintf(" dir.z %f t1 %f t2 %f tmin %f tmax %f \n", ray.direction.z, t1, t2, tmin, tmax );
   }


   bool along_x = ray.direction.x != 0.f && ray.direction.y == 0.f && ray.direction.z == 0.f ;
   bool along_y = ray.direction.x == 0.f && ray.direction.y != 0.f && ray.direction.z == 0.f ;
   bool along_z = ray.direction.x == 0.f && ray.direction.y == 0.f && ray.direction.z != 0.f ;

   bool in_x = ray.origin.x > bmin.x && ray.origin.x < bmax.x  ;
   bool in_y = ray.origin.y > bmin.y && ray.origin.y < bmax.y  ;
   bool in_z = ray.origin.z > bmin.z && ray.origin.z < bmax.z  ;

   bool valid_intersect ;
   if(     along_x) valid_intersect = in_y && in_z ;
   else if(along_y) valid_intersect = in_x && in_z ; 
   else if(along_z) valid_intersect = in_x && in_y ; 
   else             valid_intersect = ( tmax > tmin && tmax > 0.f ) ;  // segment of ray intersects box, at least one is ahead

   rtPrintf(" along_x %d along_y %d along_z %d in_x %d in_y %d in_z %d valid_intersect %d \n", along_x, along_y, along_z, in_x, in_y, in_z, valid_intersect  );

   if( valid_intersect ) 
   {
       float tint = tmin > 0.f ? tmin : tmax ;
 
       tt = tint > tt_min ? tint : tt_min ;  

       rtPrintf(" intersect_box_branching : tmin %f tmax %f tt %f tt_min %f \n", tmin, tmax, tt, tt_min  );

       float3 p = ray.origin + tt*ray.direction - bcen ; 
       float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;

       float3 n = make_float3(0.f) ;
       if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
       else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
       else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

       tt_normal = tt > tt_min ? n : tt_normal ;

   }
}


static __device__ void intersect_box_tavianator(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
   const float3 bmin = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
   const float3 bmax = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   // unrolled version of 
   //  https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

   float tmin, tmax, t1, t2 ; 


   t1 = (bmin.x - ray.origin.x)/ray.direction.x ;
   t2 = (bmax.x - ray.origin.x)/ray.direction.x ;
 
   tmin = min(t1, t2);
   tmax = max(t1, t2);

   t1 = (bmin.y - ray.origin.y)/ray.direction.y ;
   t2 = (bmax.y - ray.origin.y)/ray.direction.y ;
 
   //tmin = max(tmin, min(min(t1, t2), tmax));
   //tmax = min(tmax, max(max(t1, t2), tmin));
   tmin = max(tmin, min(t1, t2));
   tmax = min(tmax, max(t1, t2));


   t1 = (bmin.z - ray.origin.z)/ray.direction.z ;
   t2 = (bmax.z - ray.origin.z)/ray.direction.z ;
 
   //tmin = max(tmin, min(min(t1, t2), tmax));
   //tmax = min(tmax, max(max(t1, t2), tmin));
   tmin = max(tmin, min(t1, t2));
   tmax = min(tmax, max(t1, t2));


   bool valid = tmax > max(tmin, 0.f);

   float _tt =  valid 
             ? 
                ( tmin > 0.f ? tmin : tmax )
             : 
                ( tmax ) 
             ;

   tt = _tt > tt_min ? _tt : tt_min ;  

   rtPrintf(" intersect_box_tavianator : tmin %f tmax %f tt %f tt_min %f \n", tmin, tmax, tt, tt_min  );

   float3 p = ray.origin + tt*ray.direction - bcen ; 
   float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;

   float3 n = make_float3(0.f) ;
   if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
   else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
   else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

   tt_normal = tt > tt_min ? n : tt_normal ;
}


