static __device__
void csg_bounds_sphere(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    float radius = q0.f.w;
    float3 mn = make_float3( q0.f.x - radius, q0.f.y - radius, q0.f.z - radius );
    float3 mx = make_float3( q0.f.x + radius, q0.f.y + radius, q0.f.z + radius );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}

static __device__
bool csg_intersect_sphere(const quad& q0, const float& tt_min, float4& tt, const float3& ray_origin, const float3& ray_direction )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

    float3 O = ray_origin - center;
    float3 D = ray_direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);

    float disc = b*b-d*c;

    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // ray has segment within sphere for sdisc > 0.f 
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always

    // FORMER SCALING ISSUE DUE TO ASSUMPTION IN ABOVE OF NORMALIZED RAY_DIRECTION 

    float tt_cand = sdisc > 0.f ? ( root1 > tt_min ? root1 : root2 ) : tt_min ; 

    bool isect = tt_cand > tt_min ;
    if(isect)
    {        
        tt.x = (O.x + tt_cand*D.x)/radius ; 
        tt.y = (O.y + tt_cand*D.y)/radius ; 
        tt.z = (O.z + tt_cand*D.z)/radius ; 

        // x,y,z in frame with unit sphere at origin 
        // normalized by construction,  (x/r)^2 + (y/r)^2 + (z/r)^2 = 1

        tt.w = tt_cand ; 
    }
    return isect ; 
}





static __device__
void csg_bounds_box(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float hside = q0.f.w ; 
    const float3 bmin = make_float3(q0.f.x - hside, q0.f.y - hside, q0.f.z - hside ); 
    const float3 bmax = make_float3(q0.f.x + hside, q0.f.y + hside, q0.f.z + hside ); 

    Aabb tbb(bmin, bmax);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}

static __device__
bool csg_intersect_box(const quad& q0, const float& tt_min, float4& tt, const float3& ray_origin, const float3& ray_direction )
{
   const float hside = q0.f.w ; 
   const float3 bmin = make_float3(q0.f.x - hside, q0.f.y - hside, q0.f.z - hside ); 
   const float3 bmax = make_float3(q0.f.x + hside, q0.f.y + hside, q0.f.z + hside ); 

   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

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

   bool has_valid_intersect = false ; 
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

       float tt_cand = tt_min < t_near ?  t_near : ( tt_min < t_far ? t_far : tt_min ) ; 

       //rtPrintf(" intersect_box : t_near %f t_far %f tt %f tt_min %f \n", t_near, t_far, tt, tt_min  );

       float3 p = ray_origin + tt_cand*ray_direction - bcen ; 
       float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;

       float3 n = make_float3(0.f) ;
       if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
       else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
       else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

       if(tt_cand > tt_min)
       {
           has_valid_intersect = true ; 
           tt.x = n.x ;
           tt.y = n.y ;
           tt.z = n.z ;
           tt.w = tt_cand ; 
       }
   }

   return has_valid_intersect ; 
}

static __device__
void csg_intersect_part(unsigned partIdx, const float& tt_min, float4& tt  )
{
    Part pt = partBuffer[partIdx] ; 
    unsigned partType = pt.partType() ; 
    unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None

    if(gtransformIdx == 0)
    {
        switch(partType)
        {
            case CSG_SPHERE: csg_intersect_sphere(pt.q0,tt_min, tt, ray.origin, ray.direction )  ; break ; 
            case CSG_BOX:    csg_intersect_box(   pt.q0,tt_min, tt, ray.origin, ray.direction )  ; break ; 
        }
    }
    else
    {
        unsigned tIdx = 3*(gtransformIdx-1) ;  // transform
        if(tIdx + 2 >= tranBuffer.size())
        { 
            rtPrintf("##csg_intersect_part ABORT tIdx+2 %3u overflows tranBuffer.size \n", tIdx+2 );
            return ;  
        }

        //optix::Matrix4x4 T = tranBuffer[tIdx+0] ;  // transform (not used here, but needed in bbox transforming in bounds)
        optix::Matrix4x4 V = tranBuffer[tIdx+1] ;  // inverse transform 
        optix::Matrix4x4 Q = tranBuffer[tIdx+2] ;  // inverse transform transposed

        float4 origin    = make_float4( ray.origin.x, ray.origin.y, ray.origin.z, 1.f );           // w=1 for position  
        float4 direction = make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0.f );  // w=0 for vector

        origin    = origin * V ;    // world frame into primitive frame with inverse transform
        direction = direction * V ;  // <-- will loose normalization with scaling, intersects MUST NOT assume normalized ray direction

        float3 ray_origin = make_float3( origin.x, origin.y, origin.z );
        float3 ray_direction = make_float3( direction.x, direction.y, direction.z ); 

        bool valid_intersect = false ; 

        switch(partType)
        {
            case CSG_SPHERE: valid_intersect = csg_intersect_sphere(pt.q0,tt_min, tt, ray_origin, ray_direction )  ; break ; 
            case CSG_BOX:    valid_intersect = csg_intersect_box(   pt.q0,tt_min, tt, ray_origin, ray_direction )  ; break ; 
        }

        if(valid_intersect)
        {
            float4 ttn = make_float4( tt.x, tt.y, tt.z , 0.f );

            ttn = ttn * Q   ;  // primitive frame normal into world frame, using inverse transform transposed
            
            tt.x = ttn.x ; 
            tt.y = ttn.y ; 
            tt.z = ttn.z ; 
        }
    }
}
