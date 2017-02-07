

static __device__
void intersect_sphere(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
    // when an intersection is found between the ray and the sphere 
    // with parametric t greater than the tmin parameter
    // tt is set to the parametric t of the intersection
    // and corresponding tt_normal is set

    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

    float3 O = ray.origin - center;
    float3 D = ray.direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float disc = b*b-c;

    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;
    float root1 = -b - sdisc ;
    float root2 = -b + sdisc ;

    bool valid_intersect = sdisc > 0.f ;   // ray has a segment within the sphere

    if(valid_intersect)   
    {
        tt =  root1 > tt_min ? root1 : root2 ; 
        tt_normal = tt > tt_min ? (O + tt*D)/radius : tt_normal ; 
    }

}



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


static __device__
void intersect_box(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
   const float3 bmin = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
   const float3 bmax = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   float3 idir = make_float3(1.f)/ray.direction ; 

   // the below t-parameter float3 are intersects with the x, y and z planes of
   // the three axis slab planes through the box bmin and bmax  

   float3 t0 = (bmin - ray.origin)*idir;      //  intersects with bmin x,y,z slab planes
   float3 t1 = (bmax - ray.origin)*idir;      //  intersects with bmax x,y,z slab planes 

   float3 near = fminf(t0, t1);               //  bmin or bmax intersects closest to origin  
   float3 far  = fmaxf(t0, t1);               //  bmin or bmax intersects farthest from origin 

   float t_near = fmaxf( near );              //  furthest near intersect              
   float t_far  = fminf( far );               //  closest far intersect 


  // rtPrintf(" bmin %f %f %f ", bmin.x, bmin.y, bmin.z );
  /*
     rtPrintf(" ray.origin %f %f %f ray.direction %f %f %f idir %f %f %f \n ", 
           ray.origin.x,    ray.origin.y, ray.origin.z, 
           ray.direction.x, ray.direction.y, ray.direction.z,
           idir.x, idir.y,  idir.z 
       );

   rtPrintf(" idir %f %f %f t0 %f %f %f t1 %f %f %f \n",
         idir.x, idir.y, idir.z,  
         t0.x, t0.y, t0.z, 
         t1.x, t1.y, t1.z
      );

   rtPrintf(" near %f %f %f -> t_near %f   far %f %f %f -> t_far %f   \n",
         near.x, near.y, near.z, t_near,   
         far.x,  far.y,  far.z, t_far 
    );

*/

   bool along_x = ray.direction.x != 0.f && ray.direction.y == 0.f && ray.direction.z == 0.f ;
   bool along_y = ray.direction.x == 0.f && ray.direction.y != 0.f && ray.direction.z == 0.f ;
   bool along_z = ray.direction.x == 0.f && ray.direction.y == 0.f && ray.direction.z != 0.f ;

   bool in_x = ray.origin.x > bmin.x && ray.origin.x < bmax.x  ;
   bool in_y = ray.origin.y > bmin.y && ray.origin.y < bmax.y  ;
   bool in_z = ray.origin.z > bmin.z && ray.origin.z < bmax.z  ;

   bool has_intersect ;
   if(     along_x) has_intersect = in_y && in_z ;
   else if(along_y) has_intersect = in_x && in_z ; 
   else if(along_z) has_intersect = in_x && in_y ; 
   else             has_intersect = ( t_far > t_near && t_far > 0.f ) ;  // segment of ray intersects box, at least one is ahead

   if( has_intersect ) 
   {
       //  just because the ray intersects the box doesnt 
       //  mean its a usable intersect, there are 3 possibilities
       //
       //                t_near       t_far   
       //
       //                  |           |
       //        -----1----|----2------|------3---------->
       //                  |           |
       //
       //

       tt =  tt_min < t_near ?  
                              t_near 
                           :
                              ( tt_min < t_far ? t_far : tt_min )
                           ; 


       //rtPrintf(" intersect_box : t_near %f t_far %f tt %f tt_min %f \n", t_near, t_far, tt, tt_min  );

       float3 p = ray.origin + tt*ray.direction - bcen ; 
       float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;

       float3 n = make_float3(0.f) ;
       if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
       else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
       else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

       tt_normal = tt > tt_min ? n : tt_normal ;
   }
}

static __device__
IntersectionState_t intersect_part(unsigned partIdx, const float& tt_min, float3& tt_normal, float& tt  )
{
    quad q0, q2 ; 
    q0.f = partBuffer[4*partIdx+0];
    q2.f = partBuffer[4*partIdx+2];

    // Above sets boundary index from partBuffer, see npy/NPart.hpp for layout (also GPmt)
    // at intersections the uint4 identity is copied into the instanceIdentity attribute,
    // hence making it available to material1_propagate.cu:closest_hit_propagate
    // where crucially the instanceIdentity.z -> boundaryIndex


    NPart_t partType = (NPart_t)q2.i.w ; 

    tt = tt_min ; 

    switch(partType)
    {
        case SPHERE: intersect_sphere(q0,tt_min, tt_normal, tt)  ; break ; 
        case BOX:    intersect_box(   q0,tt_min, tt_normal, tt)  ; break ; 
        //case BOX:    intersect_box_tavianator(   q0,tt_min, tt_normal, tt)  ; break ; 
        //case BOX:    intersect_box_branching(   q0,tt_min, tt_normal, tt)  ; break ; 
    }

    IntersectionState_t state = tt > tt_min ? 
                                              ( dot(tt_normal, ray.direction) < 0.f ? Enter : Exit ) 
                                           :
                                              Miss
                                           ; 
    return state  ; 
}



