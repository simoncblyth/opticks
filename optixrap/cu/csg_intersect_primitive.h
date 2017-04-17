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
void csg_bounds_plane(const quad& q0, optix::Aabb* /*aabb*/, optix::Matrix4x4* /*tr*/  )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
   const float d = q0.f.w ; 
   rtPrintf("## csg_bounds_plane n %7.3f %7.3f %7.3f  d %7.3f  \n", n.x, n.y, n.z, d );
}
static __device__
bool csg_intersect_plane(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
   const float d = q0.f.w ; 

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


static __device__
void csg_bounds_slab(const quad& q0, const quad& q1, optix::Aabb* /*aabb*/, optix::Matrix4x4* /*tr*/  )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
   const float a = q1.f.x ; 
   const float b = q1.f.y ; 
   rtPrintf("## csg_bounds_slab n %7.3f %7.3f %7.3f  a %7.3f b %7.3f \n", n.x, n.y, n.z, a, b );
}

static __device__
bool csg_intersect_slab(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
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

   float t_cand = t_near > t_min ?  t_near : ( t_far > t_min ? t_far : t_min ) ; 

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





enum
{
    CYLINDER_ENDCAP_P = 0x1 <<  0,    
    CYLINDER_ENDCAP_Q = 0x1 <<  1
};    

static __device__
void csg_bounds_cylinder(const quad& q0, const quad& q1, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float3  center = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
    const float   radius = q0.f.w ; 
    const float    sizeZ = q1.f.x  ; 
    const unsigned flags = q1.u.y ;

    bool PCAP = flags & CYLINDER_ENDCAP_P ; 
    bool QCAP = flags & CYLINDER_ENDCAP_Q ;

    rtPrintf("## csg_bounds_cylinder center %7.3f %7.3f %7.3f radius %7.3f  sizeZ %7.3f flags %u PCAP %d QCAP %d \n",
          center.x, center.y, center.z, radius, sizeZ, flags, PCAP, QCAP );

    const float3 bbmin = make_float3( center.x - radius, center.y - radius, center.z - sizeZ/2.f );
    const float3 bbmax = make_float3( center.x + radius, center.y + radius, center.z + sizeZ/2.f );

    Aabb tbb(bbmin, bbmax);
    if(tr) transform_bbox( &tbb, tr );  
    aabb->include(tbb);
}

static __device__
bool csg_intersect_cylinder(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    // ascii art explanation in intersect_ztubs.h

    const float   radius = q0.f.w ; 
    const float    sizeZ = q1.f.x  ; 
    const unsigned flags = q1.u.y ;
    const float3 position = make_float3( q0.f.x, q0.f.y, q0.f.z - sizeZ/2.f ); // P: point on axis at base of cylinder

    bool PCAP = flags & CYLINDER_ENDCAP_P ; 
    bool QCAP = flags & CYLINDER_ENDCAP_Q ;

    const float3 m = ray_origin - position ;
    const float3 n = ray_direction ; 
    const float3 d = make_float3(0.f, 0.f, sizeZ );   // PQ : cylinder axis 

    float rr = radius*radius ; 
    float3 dnorm = normalize(d);  

    float mm = dot(m, m) ; 
    float nn = dot(n, n) ; 
    float dd = dot(d, d) ;  
    float nd = dot(n, d) ;
    float md = dot(m, d) ;
    float mn = dot(m, n) ; 
    float k = mm - rr ; 

    // quadratic coefficients of t,     a tt + 2b t + c = 0 
    float a = dd*nn - nd*nd ;   
    float b = dd*mn - nd*md ;
    float c = dd*k - md*md ; 

    float disc = b*b-a*c;

    // axial ray endcap handling 
    if(fabs(a) < 1e-6f)     
    {
        if(c > 0.f) return false ;  // ray starts and ends outside cylinder

        if(md < 0.f && PCAP)        // ray origin on P side
        {
            isect.x = -dnorm.x ; 
            isect.y = -dnorm.y ; 
            isect.z = -dnorm.z ; 
            isect.w = -mn/nn ;      // P endcap 
        } 
        else if(md > dd && QCAP) // ray origin on Q side 
        {
            isect.x = dnorm.x ; 
            isect.y = dnorm.y ; 
            isect.z = dnorm.z ; 
            isect.w = (nd - mn)/nn ;  // Q endcap
        }
        else    // md 0->dd, ray origin inside 
        {
            if( nd > 0.f && PCAP) // ray along +d 
            {
                isect.x = dnorm.x ; 
                isect.y = dnorm.y ; 
                isect.z = dnorm.z ; 
                isect.w = -mn/nn ;    // P endcap from inside
            } 
            else if(QCAP)  // ray along -d
            {
                isect.x = -dnorm.x ; 
                isect.y = -dnorm.y ; 
                isect.z = -dnorm.z ; 
                isect.w = (nd - mn)/nn ;  // Q endcap from inside
            }
            return false  ;   // hmm  
        }
    }  // end-of-axial-ray endcap handling 
    


    if(disc > 0.0f)  // intersection with the infinite cylinder
    {
        float sdisc = sqrtf(disc);

        float root1 = (-b - sdisc)/a;     
        float ad1 = md + root1*nd ;        // axial coord of intersection point 
        float3 P1 = ray_origin + root1*ray_direction ;  

        if( ad1 > 0.f && ad1 < dd )  // intersection inside cylinder range
        {
            float3 N  = (P1 - position)/radius  ;  
            N.z = 0.f ; 
            N = normalize(N);

            isect.x = N.x ; 
            isect.y = N.y ; 
            isect.z = N.z ; 
            isect.w = root1 ; 
            // HP_WALL_O ;
        } 
        else if( ad1 < 0.f && PCAP ) //  intersection outside cylinder on P side
        {
            if( nd <= 0.f ) return false ; // ray direction away from endcap
            float t = -md/nd ;   // P endcap 
            float checkr = k + t*(2.f*mn + t*nn) ; // bracket typo in book 2*t*t makes no sense   
            if ( checkr < 0.f )
            {
                isect.x = -dnorm.x ; 
                isect.y = -dnorm.y ; 
                isect.z = -dnorm.z ; 
                isect.w = t ; 
                // HP_PCAP_O ;
            } 
        } 
        else if( ad1 > dd && QCAP  ) //  intersection outside cylinder on Q side
        {
            if( nd >= 0.f ) return false ; // ray direction away from endcap
            float t = (dd-md)/nd ;   // Q endcap 
            float checkr = k + dd - 2.0f*md + t*(2.f*(mn-nd)+t*nn) ;             
            if ( checkr < 0.f )
            {
                isect.x = dnorm.x ; 
                isect.y = dnorm.y ; 
                isect.z = dnorm.z ; 
                isect.w = t ; 
                // HP_QCAP_O ;
            } 
        }

        float root2 = (-b + sdisc)/a;     // far root : means are inside (always?)
        float ad2 = md + root2*nd ;        // axial coord of far intersection point 
        float3 P2 = ray.origin + root2*ray.direction ;  

        if( ad2 > 0.f && ad2 < dd )  // intersection from inside against wall 
        {
            float3 N  = (P2 - position)/radius  ;  
            N.z = 0.f ; 
            N = -normalize(N);

            isect.x = N.x ; 
            isect.y = N.y ; 
            isect.z = N.z ; 
            isect.w = root2 ; 
            // HP_WALL_I ;
        } 
        else if( ad2 < 0.f && PCAP ) //  intersection from inside to P endcap
        {
            float t = -md/nd ;   // P endcap 
            float checkr = k + t*(2.f*mn + t*nn) ; // bracket typo in book 2*t*t makes no sense   
            if ( checkr < 0.f )
            {
                isect.x = dnorm.x ; 
                isect.y = dnorm.y ; 
                isect.z = dnorm.z ; 
                isect.w = t  ; 
                // HP_PCAP_I 
            } 
        } 
        else if( ad2 > dd  && QCAP ) //  intersection from inside to Q endcap
        {
            float t = (dd-md)/nd ;   // Q endcap 
            float checkr = k + dd - 2.0f*md + t*(2.f*(mn-nd)+t*nn) ;             
            if ( checkr < 0.f )
            {
                isect.x = -dnorm.x ; 
                isect.y = -dnorm.y ; 
                isect.z = -dnorm.z ; 
                isect.w = t  ; 
                // HP_QCAP_I ;
            } 
        }
    }
    return true ; 
}



