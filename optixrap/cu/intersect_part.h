

static __device__
void intersect_sphere(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
    // when an intersection is found between the ray and the sphere 
    // with parametric t greater than the tmin parameter
    // the tt set to the parametric t found
    // 

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

    tt =  root1 > tt_min && sdisc > 0.f ? 
                                         ( root1 )
                                      :
                                         ( root2 > tt_min && sdisc > 0.f  ? root2 : tt_min )  
                                      ; 

    tt_normal = tt > tt_min ? (O + tt*D)/radius : tt_normal ; 
}


static __device__
void intersect_box(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
{
   const float3 bmin = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
   const float3 bmax = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   float3 idir = make_float3(1.f)/ray.direction ; 
   float3 t0 = (bmin - ray.origin)*idir;
   float3 t1 = (bmax - ray.origin)*idir;


  // rtPrintf(" bmin %f %f %f ", bmin.x, bmin.y, bmin.z );

   // https://tavianator.com/fast-branchless-raybounding-box-intersections/
   // https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

   float3 near = fminf(t0, t1);
   float3 far = fmaxf(t0, t1);

   float tmin = fmaxf( near );
   float tmax = fminf( far );

/*
   rtPrintf(" idir %f %f %f t0 %f %f %f t1 %f %f %f \n",
         idir.x, idir.y, idir.z,  
         t0.x, t0.y, t0.z, 
         t1.x, t1.y, t1.z
      );
 idir inf inf -1.000000 t0 -inf -inf 699.000000 t1 inf inf 499.000000 
 idir inf inf -1.000000 t0 -inf -inf 699.000000 t1 inf inf 499.000000 
 idir inf inf -1.000000 t0 -inf -inf 699.000000 t1 inf inf 499.000000 
 idir inf inf -1.000000 t0 -inf -inf 699.000000 t1 inf inf 499.000000 

*/

/*
   rtPrintf(" near %f %f %f -> tmin %f   far %f %f %f -> tmax %f   \n",
         near.x, near.y, near.z, tmin,   
         far.x,  far.y,  far.z, tmax
    );

 near -inf -inf 499.000000 -> tmin 499.000000   far inf inf 699.000000 -> tmax 699.000000   
 near -inf -inf 499.000000 -> tmin 499.000000   far inf inf 699.000000 -> tmax 699.000000   
 near -inf -inf 499.000000 -> tmin 499.000000   far inf inf 699.000000 -> tmax 699.000000   
 near -inf -inf 499.000000 -> tmin 499.000000   far inf inf 699.000000 -> tmax 699.000000  
*/




  // bool valid = tmin <= tmax && tmax > 0.f ;
   bool valid = tmax > fmaxf(tmin, 0.f ) ; 

   float _tt =  valid 
             ? 
                ( tmin > 0.f ? tmin : tmax )
             : 
                ( tmax ) 
             ;

   tt = _tt > tt_min ? _tt : tt_min ;  

   rtPrintf(" intersect_box : tmin %f tmax %f tt %f tt_min %f \n", tmin, tmax, tt, tt_min  );

   float3 p = ray.origin + tt*ray.direction - bcen ; 
   float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;

   float3 n = make_float3(0.f) ;
   if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
   else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
   else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

   tt_normal = tt > tt_min ? n : tt_normal ;
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
    }

    IntersectionState_t state = tt > tt_min ? 
                                              ( dot(tt_normal, ray.direction) < 0.f ? Enter : Exit ) 
                                           :
                                              Miss
                                           ; 
    return state  ; 
}



