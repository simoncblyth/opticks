

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
   const float3 min_ = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
   const float3 max_ = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
   const float3 cen_ = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   float3 t0 = (min_ - ray.origin)/ray.direction;
   float3 t1 = (max_ - ray.origin)/ray.direction;

   // slab method  :google:`tavianator`
   float3 near = fminf(t0, t1);
   float3 far = fmaxf(t0, t1);
   float tmin = fmaxf( near );
   float tmax = fminf( far );

   tt =  tmin <= tmax && tmax > 0.f && tmin > tt_min 
             ? 
                ( tmin )
             : 
                ( tmax > tt_min ? tmax : tt_min ) 
             ;

   float3 p = ray.origin + tt*ray.direction - cen_ ; 

   float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
   //float pmax = fmaxf(pa);

   float3 n = make_float3(0.f) ;
   if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
   else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
   else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

   tt_normal = n ;
}

static __device__
IntersectionState_t intersect_part(unsigned partIdx, const float& tt_min, float3& tt_normal, float& tt  )
{
    quad q0, q2 ; 
    q0.f = partBuffer[4*partIdx+0];
    q2.f = partBuffer[4*partIdx+2];

    NPart_t partType = (NPart_t)q2.i.w ; 

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



