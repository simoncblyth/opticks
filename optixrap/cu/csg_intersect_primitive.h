// csg_intersect_primitive.h : is included into top level program: intersect_analytic.cu
// csg_intersect_part.h      : is primary user of these functions

#include "robust_quadratic_roots.h"

//#include "Roots3And4.h"
#include "Vecgeom_Solve.h"


rtBuffer<float4> planBuffer ;

static __device__
void csg_bounds_convexpolyhedron(const unsigned& planeOffset, const Part& pt, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const quad& q2 = pt.q2 ; 
    const quad& q3 = pt.q3 ; 

    unsigned planeIdx = pt.planeIdx() ; 
    unsigned planeNum = pt.planeNum() ; 

    rtPrintf("## csg_bounds_convexpolyhedron planeIdx %u planeNum %u planeOffset %u  \n", planeIdx, planeNum, planeOffset );
    unsigned planeBase = planeIdx-1+planeOffset ;

    for(unsigned i=0 ; i < planeNum ; i++)
    {
        float4 plane = planBuffer[planeBase+i] ;
        rtPrintf("## csg_bounds_convexpolyhedron plane i:%u plane: %10.3f %10.3f %10.3f %10.3f  \n", i, plane.x, plane.y, plane.z, plane.w );
    } 

    float3 mn = make_float3( q2.f.x, q2.f.y,  q2.f.z );
    float3 mx = make_float3( q3.f.x, q3.f.y,  q3.f.z );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}





static __device__
bool csg_intersect_convexpolyhedron(const unsigned& planeOffset, const Part& pt, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    unsigned planeIdx = pt.planeIdx() ; 
    unsigned planeNum = pt.planeNum() ; 
    unsigned planeBase = planeIdx-1+planeOffset ;


#ifdef CSG_INTERSECT_CONVEXPOLYHEDRON_TEST
    const float3& o = ray_origin ;
    const float3& d = ray_direction ;
    rtPrintf("\n## csg_intersect_convexpolyhedron planeIdx %u planeNum %u planeOffset %u planeBase %u  \n", planeIdx, planeNum, planeOffset, planeBase );
    rtPrintf("## csg_intersect_convexpolyhedron o: %10.3f %10.3f %10.3f  d: %10.3f %10.3f %10.3f \n", o.x, o.y, o.z, d.x, d.y, d.z );
#endif

    float t0 = -CUDART_INF_F ; 
    float t1 =  CUDART_INF_F ; 

    float3 t0_normal = make_float3(0.f);
    float3 t1_normal = make_float3(0.f);

    //for(unsigned i=0 ; i < planeNum && t0 < t1  ; i++)
    for(unsigned i=0 ; i < planeNum ; i++)
    {
        float4 plane = planBuffer[planeBase+i];  
        float3 n = make_float3(plane);
        float dplane = plane.w ;

         // RTCD p199,  
         //            n.X = dplane
         //
         //             n.(o+td) = dplane
         //            no + t nd = dplane
         //                    t = (dplane - no)/nd
         //

        float nd = dot(n, ray_direction); // -ve: entering, +ve exiting halfspace  
        float no = dot(n, ray_origin ) ;  //  distance from coordinate origin to ray origin in direction of plane normal 
        float dist = no - dplane ;        //  subtract plane distance from origin to get signed distance from plane, -ve inside 
        float t_cand = -dist/nd ;

        bool parallel_inside = nd == 0.f && dist < 0.f ;   // ray parallel to plane and inside halfspace
        bool parallel_outside = nd == 0.f && dist > 0.f ;  // ray parallel to plane and outside halfspac

#ifdef CSG_INTERSECT_CONVEXPOLYHEDRON_TEST
        rtPrintf("## csg_intersect_convexpolyhedron i: %2d n: %10.3f %10.3f %10.3f dplane:%10.3f nd:%10.3f no:%10.3f  dist:%10.3f parallel_inside:%d parallel_outside:%d  t_cand: %10.3f \n", i,
          n.x, n.y, n.z, dplane, nd, no, dist, parallel_inside, parallel_outside,   t_cand );
#endif

        if(parallel_inside) continue ;       // continue to next plane 
        if(parallel_outside) return false ;  // <-- without early exit, this still works due to infinity handling 

        //    NB ray parallel to plane and outside halfspace 
        //         ->  t_cand = -inf 
        //                 nd = 0.f 
        //                t1 -> -inf  
        //          hence t0 > t1 and get no intersect 
        //

        if( nd < 0.f)  // entering 
        {
            if(t_cand > t0)
            {
                t0 = t_cand ;
                t0_normal = n ;
            }
        } 
        else     // exiting
        {       
            if(t_cand < t1)
            {
                t1 = t_cand ;
                t1_normal = n ;
            }
        }
    }

#ifdef CSG_INTERSECT_CONVEXPOLYHEDRON_TEST
    rtPrintf("## csg_intersect_convexpolyhedron t0:%10.3f t1: %10.3f \n", t0, t1 );
#endif


    bool valid_intersect = t0 < t1 ; 
    if(valid_intersect)
    {
        if( t0 > t_min )
        {
            isect.x = t0_normal.x ; 
            isect.y = t0_normal.y ; 
            isect.z = t0_normal.z ; 
            isect.w = t0 ; 
        } 
        else if( t1 > t_min )
        {
            isect.x = t1_normal.x ; 
            isect.y = t1_normal.y ; 
            isect.z = t1_normal.z ; 
            isect.w = t1 ; 
        }
    }
    return valid_intersect ;
}


static __device__
void csg_intersect_convexpolyhedron_test(unsigned long long photon_id)
{
    unsigned planeOffset = 0 ; 
    Part pt ; 
    pt.setPlaneIdx(1) ; 
    pt.setPlaneNum(6) ; 

    float t_min = 0.f ; 
    float3 ray_direction = make_float3(    0.f, 0.f, -1.f );
    float3 ray_origin    = make_float3(    0.f, 0.f, 500.f ); // off the side of the box

    for(unsigned i=0 ; i < 2 ; i++)
    {
        ray_origin.x = i == 0 ? 0.f : 300.f ;    //   mid-cube or off the side
        bool expect_has_isect = i == 0 ; 

        float4 isect = make_float4(0.f,0.f,0.f,0.f);
        bool has_isect = csg_intersect_convexpolyhedron(planeOffset, pt, t_min , isect, ray_origin, ray_direction );

        float t = isect.w ;  
        float3 p = ray_origin + t*ray_direction ; 

        rtPrintf("## csg_intersect_convexpolyhedron_test pid:%llu i:%u has_isect:%d isect:(%10.3f %10.3f %10.3f %10.3f) p:(%10.3f %10.3f %10.3f) \n",
                     photon_id, i, has_isect, isect.x, isect.y, isect.z, isect.w, p.x, p.y, p.z ); 

        if(has_isect != expect_has_isect) rtPrintf("## csg_intersect_convexpolyhedron_test FAILURE \n"); 
    }

}




static __device__
void csg_bounds_cone(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    float r1 = q0.f.x ; 
    float z1 = q0.f.y ; 
    float r2 = q0.f.z ; 
    float z2 = q0.f.w ; 

    float rmax = fmaxf(r1, r2) ;

    float tan_theta = (r2-r1)/(z2-z1) ;
    float z_apex = (z2*r1-z1*r2)/(r1-r2) ; 
 
    rtPrintf("## csg_bounds_cone r1:%10.3f z1:%10.3f r2:%10.3f z2:%10.3f rmax:%10.3f tan_theta:%10.3f z_apex:%10.3f  \n", r1,z1,r2,z2, rmax, tan_theta, z_apex) ; 

    if(z2 < z1) rtPrintf("## csg_bounds_cone z2 < z1, z1: %10.3f z2 %10.3f \n", z1, z2 );

    float3 mx = make_float3(  rmax,  rmax,  z2 );
    float3 mn = make_float3( -rmax, -rmax,  z1 );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}


enum {
    NEAR_CAP  = 0x1 << 0 ,
    NEAR_OBJ  = 0x1 << 1 ,
    FAR_OBJ   = 0x1 << 2 ,
    FAR_CAP   = 0x1 << 3 
};


static __device__
bool csg_intersect_cone(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    float r1 = q0.f.x ; 
    float z1 = q0.f.y ; 
    float r2 = q0.f.z ; 
    float z2 = q0.f.w ;   // z2 > z1

    float tth = (r2-r1)/(z2-z1) ;
    float tth2 = tth*tth ; 
    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex
 
    float r1r1 = r1*r1 ; 
    float r2r2 = r2*r2 ; 

#ifdef CSG_INTERSECT_CONE_TEST
    rtPrintf("## csg_intersect_cone r1:%10.3f z1:%10.3f r2:%10.3f z2:%10.3f tth:%10.3f tth2:%10.3f z0:%10.3f r1r1:%10.3f r2r2:%10.3f \n",
          r1,z1,r2,z2,tth,tth2,z0,r1r1,r2r2);
#endif

    const float3& o = ray_origin ;
    const float3& d = ray_direction ;

#ifdef CSG_INTERSECT_CONE_TEST
    rtPrintf("## csg_intersect_cone o: %10.3f %10.3f %10.3f  d: %10.3f %10.3f %10.3f \n", o.x, o.y, o.z, d.x, d.y, d.z );
#endif

   //  cone with apex at [0,0,z0]  and   r1/(z1-z0) = tanth  for any r1,z1 on the cone
   //
   //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
   //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
   //
   //   Gradient:    [2x, 2y, (-2z tanth^2) + 2z0 tanth^2 ] 
   //
   //   (o.x+ t d.x)^2 + (o.y + t d.y)^2 - (o.z - z0 + t d.z)^2 tth2 = 0 
   // 
   // quadratic in t :    c2 t^2 + 2 c1 t + c0 = 0 


    float c2 = d.x*d.x + d.y*d.y - d.z*d.z*tth2 ;
    float c1 = o.x*d.x + o.y*d.y - (o.z-z0)*d.z*tth2 ; 
    float c0 = o.x*o.x + o.y*o.y - (o.z-z0)*(o.z-z0)*tth2 ;
    float disc = c1*c1 - c0*c2 ; 

#ifdef CSG_INTERSECT_CONE_TEST
    rtPrintf("## csg_intersect_cone c2 %10.3f c1 %10.3f c0 %10.3f disc %10.3f \n", c2, c1, c0, disc );
#endif

    // * cap intersects (including axial ones) will always have potentially out of z-range cone intersects 
    // * cone intersects will have out of r-range plane intersects, other than rays within xy plane
   
 
    bool valid_isect = false ;
 
    if(disc > 0.f)  // has intersects with infinite cone
    {
        float sdisc = sqrtf(disc) ;   
        float root1 = (-c1 - sdisc)/c2 ;
        float root2 = (-c1 + sdisc)/c2 ;  
        float root1p = root1 > t_min ? root1 : RT_DEFAULT_MAX ;   // disqualify -ve roots from mirror cone immediately 
        float root2p = root2 > t_min ? root2 : RT_DEFAULT_MAX ; 

        float t_near = fminf( root1p, root2p );
        float t_far  = fmaxf( root1p, root2p );  
        float z_near = o.z+t_near*d.z ; 
        float z_far  = o.z+t_far*d.z ; 


        t_near = z_near > z1 && z_near < z2  && t_near > t_min ? t_near : RT_DEFAULT_MAX ; // disqualify out-of-z
        t_far  = z_far  > z1 && z_far  < z2  && t_far  > t_min ? t_far  : RT_DEFAULT_MAX ; 

#ifdef CSG_INTERSECT_CONE_TEST
        rtPrintf("## csg_intersect_cone sdisc %10.3f root1 %10.3f root2 %10.3f t_near %10.3f t_far %10.3f z_near %10.3f z_far %10.3f \n", sdisc, root1, root2, t_near, t_far, z_near, z_far  );
#endif

        float idz = 1.f/d.z ; 
        float t_cap1 = d.z == 0.f ? RT_DEFAULT_MAX : (z1 - o.z)*idz ;   // d.z zero means no z-plane intersects
        float t_cap2 = d.z == 0.f ? RT_DEFAULT_MAX : (z2 - o.z)*idz ;
        float r_cap1 = (o.x + t_cap1*d.x)*(o.x + t_cap1*d.x) + (o.y + t_cap1*d.y)*(o.y + t_cap1*d.y) ;  
        float r_cap2 = (o.x + t_cap2*d.x)*(o.x + t_cap2*d.x) + (o.y + t_cap2*d.y)*(o.y + t_cap2*d.y) ;  

        t_cap1 = r_cap1 < r1r1 && t_cap1 > t_min ? t_cap1 : RT_DEFAULT_MAX ;  // disqualify out-of-radius
        t_cap2 = r_cap2 < r2r2 && t_cap2 > t_min ? t_cap2 : RT_DEFAULT_MAX ; 
 
        float t_capn = fminf( t_cap1, t_cap2 );    // order caps
        float t_capf = fmaxf( t_cap1, t_cap2 );


        // SIMPLE WAY fminf(float4) : appears to work for single cone, but 
        //  seeing issues when combine such cones within a union
        //  with wierd mirror cone cutoffs as rotate around 
        //  ... that was not using RT_DEFAULT_MAX 
        //  
        float4 tt = make_float4( t_near, t_far, t_capn, t_capf );
        float t_cand = fminf(tt) ; 
        
/*
        float t_cand = t_min ; 
        bool near_cap  = t_capn > t_min && t_capn < RT_DEFAULT_MAX ;
        bool far_cap   = t_capf > t_min && t_capf < RT_DEFAULT_MAX ;
        bool near_cone = t_near > t_min && t_near < RT_DEFAULT_MAX ;
        bool far_cone  = t_far  > t_min && t_far  < RT_DEFAULT_MAX ;

        unsigned code = 0u ; 
        if(near_cap)  code |=  NEAR_CAP  ; 
        if(near_cone) code |=  NEAR_OBJ ; 
        if(far_cone)  code |=  FAR_OBJ  ; 
        if(far_cap)   code |=  FAR_CAP  ; 
         
        switch(code)
        { 
            case                0u:   t_cand = t_min                   ; break ; 
            case          NEAR_CAP:   t_cand = t_capn                  ; break ;
            case          NEAR_OBJ:   t_cand = t_near                  ; break ;
            case           FAR_OBJ:   t_cand = t_far                   ; break ;
            case           FAR_CAP:   t_cand = t_capf                  ; break ;
            case NEAR_CAP|NEAR_OBJ:   t_cand = fminf( t_capn, t_near ) ; break ;
            case  NEAR_CAP|FAR_OBJ:   t_cand = fminf( t_capn, t_far)   ; break ;
            case   FAR_CAP|FAR_OBJ:   t_cand = fminf( t_capf, t_far )  ; break ;
            case  NEAR_OBJ|FAR_OBJ:   t_cand = fminf( t_near, t_far)   ; break ;
            case  NEAR_CAP|FAR_CAP:   t_cand = fminf( t_capf, t_capn)  ; break ;

            default:
                 rtPrintf("## unhandled combi %x \n", code) ; break ; 
        }
*/


        // HMM: 
        //
        // * cannot just pick the closest, as if one fails need to fall back to next 
        //   however the correct priority order depends on the disposition of the ray
        //
        // * This approach really powerful way to build case logic by experiment (ie in incremental way) 
        //   because it splayed the logical possibilities out into a long line of 16 possible combinations
        //
        // * Note that the ordering of the switch does not matter... as the code
        //   composed of the 4 input bits can be combined in 16 possible ways from 0000 to 1111 
        //

        valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
        if(valid_isect)
        {
            //rtPrintf("## csg_intersect_cone t_min:%10.3f t_cand:%10.3f    t_cap1:%10.3f t_cap2:%10.3f t_near:%10.3f t_far:%10.3f \n", t_min, t_cand, t_cap1, t_cap2, t_near, t_far );
            if( t_cand == t_cap1 || t_cand == t_cap2 )
            {
                isect.x = 0.f ; 
                isect.y = 0.f ;
                isect.z =  t_cand == t_cap2 ? 1.f : -1.f  ;   
            }
            else
            { 

               //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
               //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
               //
               //   Gradient:    [2x, 2y, (-2z + 2z0) tanth^2 ] 
               //   Gradient:    2*[x, y, (z0-z) tanth^2 ] 


                float3 n = normalize(make_float3( o.x+t_cand*d.x, o.y+t_cand*d.y, (z0-(o.z+t_cand*d.z))*tth2  ))  ; 
#ifdef CSG_INTERSECT_CONE_TEST
                float3 p = make_float3( o.x+t_cand*d.x, o.y+t_cand*d.y, o.z+t_cand*d.z) ; 
                float tth2_check = (p.x*p.x + p.y*p.y)/((p.z-z0)*(p.z-z0)) ;
                float tth2_delta = tth2 - tth2_check ; 
                float3 v = normalize(make_float3( p.x , p.y , p.z - z0 ));   // direction vector from apex (0,0,z0) to p
                float vn = dot(v,n);  
                rtPrintf("## csg_intersect_cone c2 %10.3f  p %10.3f %10.3f %10.3f   n %10.3f %10.3f %10.3f  v %10.3f %10.3f %10.3f  vn %10.3f   tth2_delta %10.3f \n", 
                      c2, p.x, p.y, p.z, n.x, n.y, n.z, v.x, v.y, v.z, vn,   tth2_delta );
#endif
                isect.x = n.x ; 
                isect.y = n.y ;
                isect.z = n.z ; 
            }
            isect.w = t_cand ; 
        }
    }
    return valid_isect ; 
}
 




static __device__
unsigned classify_x( const float x, const float r1, const float r2)
{ 
   // hmm : this approach obfuscates expectations 
    unsigned i =  0 ; 
    if(      x < -r1 )             i = 0 ;   
    else if( x > -r1 && x < -r2 )  i = 1 ; 
    else if( x > -r2 && x < r2  )  i = 2 ; 
    else if( x > r2  && x < r1  )  i = 3 ; 
    else if( x > r1  )             i = 4 ;
    return i ; 
}


static __device__
void csg_intersect_cone_test(unsigned long long photon_id)
{
     /*


      0------1----2----A---3----4----5------           
                      / \
                     /   \
                    /     \
                   /       \
                  /         \
                 +-----------+         z = z2
                /     r2      \
               /               \
            t_near              \
             /                   \
            +----------|-----------+   z = z1
      [-300,0,-300]    r1       [+300,0,-300]


     */

    float r2 = 100.f ;
    float r1 = 300.f ;
    float rd = 50.f ; 

    float z2 = -100.f ;   
    float z1 = -300.f ; 
    float zd = 50.f ; 

    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex
    

    rtPrintf("## csg_intersect_cone_test z0:%10.3f \n", z0 );

    if(z1 > z2) rtPrintf("## csg_intersect_cone_test ERROR: z1>z2  (fundamental error) \n");
    if(r2 > r1) rtPrintf("## csg_intersect_cone_test ERROR: r2>r1  (violates assumption used in this test) \n");

    quad q0 ;  
    q0.f.x = r1 ; 
    q0.f.y = z1 ; 
    q0.f.z = r2 ; 
    q0.f.w = z2 ; 

    float t_min = 0.f ; 

    // shoot down cone axis (-z) at various x along y=0 line in xy plane
    float3 ray_direction = make_float3( 0.f, 0.f, 0.f );
    float3 ray_origin    = make_float3( 0.f ,0.f, 0.f );

    for(unsigned j=0 ; j < 2 ; j++)
    {
        float dz = 0.f ; 
        float line_z = 0.f ; 

        switch(j)
        {
            case 0:dz = -1.f ; line_z = z0      ; break ; 
            case 1:dz =  1.f ; line_z = z1 - zd ; break ; 
        }
        ray_direction.z = dz ; 

        for(unsigned i=0 ; i < 6 ; i++)
        {
            float x = 0.f ; 
            switch(i)
            {
                case 0:x = -(r1+rd) ; break ; 
                case 1:x = -(r1-rd) ; break ; 
                case 2:x = -(r2-rd) ; break ; 
                case 3:x =  (r2-rd) ; break ; 
                case 4:x =  (r1-rd) ; break ; 
                case 5:x =  (r1+rd) ; break ; 
            }
            ray_origin.x = x ; 
            ray_origin.y = 0.f ; 
            ray_origin.z = line_z ; 

            float4 isect = make_float4(0.f, 0.f, 0.f, 0.f );

            bool valid_isect = csg_intersect_cone(q0, t_min , isect, ray_origin, ray_direction );

            float t = isect.w ;  
            float3 p = ray_origin + t*ray_direction ; 

            rtPrintf("## csg_intersect_cone_test j:%d i:%d x:%10.3f valid_isect:%d isect:(%10.3f %10.3f %10.3f %10.3f) p:(%10.3f %10.3f %10.3f) \n",
                 j,i,x, valid_isect, isect.x, isect.y, isect.z, isect.w, p.x, p.y, p.z ); 

            bool expect_valid_isect = false ; 
            float expect_z = 0.f ; 

            if( j == 0)
            {
                // rays down -z from apex plane expecting: miss, cone, endcap, cone, miss

                if( x < -r1 )
                {
                    expect_valid_isect = false ; 
                    expect_z = ray_origin.z ; 
                }
                else if( x > -r1 && x < -r2 )
                {
                    expect_valid_isect = true ; 
                    expect_z = x ; 
                } 
                else if( x > -r2 && x < r2 )
                {
                    expect_valid_isect = true ; 
                    expect_z = z2 ; 
                }
                else if( x > r2  && x < r1 )
                {
                    expect_valid_isect = true ; 
                    expect_z = -x ; 
                }
                else if(x > r1)
                {
                    expect_valid_isect = false ; 
                    expect_z = ray_origin.z ; 
                } 
            }
            else if( j == 1)
            {
                // rays up +z from below z=z1 expecting: miss, endcap, endcap, endcap, miss

                if( x < -r1 )
                {
                    expect_valid_isect = false ; 
                    expect_z = ray_origin.z ; 
                }
                else if( x > -r1 && x < -r2 )
                {
                    expect_valid_isect = true ; 
                    expect_z = z1 ; 
                } 
                else if( x > -r2 && x < r2 )
                {
                    expect_valid_isect = true ; 
                    expect_z = z1 ; 
                }
                else if( x > r2  && x < r1 )
                {
                    expect_valid_isect = true ; 
                    expect_z = z1 ; 
                }
                else if(x > r1)
                {
                    expect_valid_isect = false ; 
                    expect_z = ray_origin.z ; 
                } 
            } 

            if(expect_valid_isect != valid_isect) rtPrintf("## ERROR valid_isect \n"); 
            if(fabsf(expect_z - p.z) > 1e-4f ) rtPrintf("## ERROR z %10.3f expect: %10.3f \n", p.z, expect_z );
        }
    }
} 






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
bool csg_intersect_sphere(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
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

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ; 

    bool valid_isect = t_cand > t_min ;
    if(valid_isect)
    {        
        isect.x = (O.x + t_cand*D.x)/radius ;   // normalized by construction
        isect.y = (O.y + t_cand*D.y)/radius ; 
        isect.z = (O.z + t_cand*D.z)/radius ; 
        isect.w = t_cand ; 
    }
    return valid_isect ; 
}





static __device__
void csg_bounds_zsphere(const quad& q0, const quad& q1, const quad& q2, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float radius = q0.f.w;
    const float zmax = q0.f.z + q1.f.y ;   
    const float zmin = q0.f.z + q1.f.x ;   

    const unsigned flags = q2.u.x ;
    const bool QCAP = flags & ZSPHERE_QCAP ;  
    const bool PCAP = flags & ZSPHERE_PCAP ; 

    rtPrintf("## csg_bounds_zsphere  zmin %7.3f zmax %7.3f flags %u QCAP(zmin) %d PCAP(zmax) %d  \n", zmin, zmax, flags, QCAP, PCAP );

    float3 mx = make_float3( q0.f.x + radius, q0.f.y + radius, zmax );
    float3 mn = make_float3( q0.f.x - radius, q0.f.y - radius, zmin );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}





/*
Plane eqn in general frame:
               point_in_plane.plane_normal = plane_dist_to_origin

Ray-Plane intersection 

    ( ray_origin + t ray_direction ).plane_normal = plane_dist_to_origin
 
     t = plane_dist_to_origin - ray_origin.plane_normal
        -------------------------------------------------
                ray_direction.plane_normal


Now consider plane normal to be +z axis and 

     t = plane_dist_to_origin - ray_origin.z
        --------------------------------------
              ray_direction.z

plane_dist_to_orign = zmin or zmax



Intersect with sphere

     O = ray_origin - center 
     D = ray_direction  

     (O + t D).(O + t D) = rr

   t^2 D.D + 2 t O.D + O.O - rr  = 0 

     d = D.D
     b = O.D
     c = O.O - rr

*/


static __device__
bool csg_intersect_zsphere(const quad& q0, const quad& q1, const quad& q2, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    const float3 center = make_float3(q0.f);
    float3 O = ray_origin - center;  
    float3 D = ray_direction;
    const float radius = q0.f.w;

    float b = dot(O, D);               // t of closest approach to sphere center
    float c = dot(O, O)-radius*radius; // < 0. indicates ray_origin inside sphere
    if( c > 0.f && b > 0.f ) return false ;   

    // Cannot intersect when ray origin outside sphere and direction away from sphere.
    // Whether early exit speeds things up is another question ... 

    //const unsigned flags = q2.u.x ;
    //const bool QCAP = flags & ZSPHERE_QCAP ; 
    //const bool PCAP = flags & ZSPHERE_PCAP ;  

    const bool QCAP = true ; 
    const bool PCAP = true ;  

    const float2 zdelta = make_float2(q1.f);
    const float zmax = center.z + zdelta.y ; 
    const float zmin = center.z + zdelta.x ;   

    float d = dot(D, D);               // NB NOT assuming normalized ray_direction

    float t1sph, t2sph, disc, sdisc ;   
    robust_quadratic_roots(t1sph, t2sph, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

/*
    float disc = b*b-d*c;
    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // ray has segment within sphere for sdisc > 0.f 

#ifdef ZSPHERE_NUMERICALLY_UNSTABLY
    float t1sph = (-b - sdisc)/d ;
    float t2sph = (-b + sdisc)/d ;  // t2sph > t1sph always, sdisc and d always +ve
#else
    // picking robust quadratic roots that avoid catastrophic subtraction 
    float q = b > 0.f ? -(b + sdisc) : -(b - sdisc) ; 
    float root1 = q/d  ; 
    float root2 = c/q  ;
    float t1sph = fminf( root1, root2 );
    float t2sph = fmaxf( root1, root2 );
#endif
*/


    float z1sph = ray_origin.z + t1sph*ray_direction.z ;  // sphere z intersects
    float z2sph = ray_origin.z + t2sph*ray_direction.z ; 

    float idz = 1.f/ray_direction.z ; 
    float t_QCAP = QCAP ? (zmax - ray_origin.z)*idz : t_min ;   // cap intersects,  t_min for cap not enabled
    float t_PCAP = PCAP ? (zmin - ray_origin.z)*idz : t_min ;

    float t1cap = fminf( t_QCAP, t_PCAP ) ;   // order cap intersects along the ray 
    float t2cap = fmaxf( t_QCAP, t_PCAP ) ;   // t2cap > t1cap 

    // disqualify plane intersects outside sphere t range
    if(t1cap < t1sph || t1cap > t2sph) t1cap = t_min ; 
    if(t2cap < t1sph || t2cap > t2sph) t2cap = t_min ; 

    // hmm somehow is seems unclean to have to use both z and t language

    float t_cand = t_min ; 
    if(sdisc > 0.f)
    {
        if(      t1sph > t_min && z1sph > zmin && z1sph < zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
        else if( t1cap > t_min )                                  t_cand = t1cap ;  // t1cap qualifies -> t1cap 
        else if( t2cap > t_min )                                  t_cand = t2cap ;  // t2cap qualifies -> t2cap
        else if( t2sph > t_min && z2sph > zmin && z2sph < zmax)   t_cand = t2sph ;  // t2sph qualifies and t2cap disabled or disqialified -> t2sph

        //rtPrintf("csg_intersect_zsphere t_min %7.3f t1sph %7.3f t1cap %7.3f t2cap %7.3f t2sph %7.3f t_cand %7.3f \n", t_min, t1sph, t1cap, t2cap, t2sph, t_cand ); 
    }

    bool valid_isect = t_cand > t_min ;

    if(valid_isect)
    {        
        isect.w = t_cand ; 
        if( t_cand == t1sph || t_cand == t2sph)
        {
            isect.x = (O.x + t_cand*D.x)/radius ; // normalized by construction
            isect.y = (O.y + t_cand*D.y)/radius ; 
            isect.z = (O.z + t_cand*D.z)/radius ; 
        }
        else
        {
            isect.x = 0.f ; 
            isect.y = 0.f ; 
            isect.z = t_cand == t_PCAP ? -1.f : 1.f ;  
        }
    }
    return valid_isect ; 
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
void csg_bounds_box3(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float3 bmin = make_float3(-q0.f.x/2.f, -q0.f.y/2.f, -q0.f.z/2.f ); 
    const float3 bmax = make_float3( q0.f.x/2.f,  q0.f.y/2.f,  q0.f.z/2.f ); 

    Aabb tbb(bmin, bmax);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}



static __device__
bool _csg_intersect_box(const float3& bmin, const float3& bmax, const float3& bcen, const float& tt_min, float4& tt, const float3& ray_origin, const float3& ray_direction )
{
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

       float3 pa = make_float3(fabs(p.x)/(bmax.x - bmin.x), 
                               fabs(p.y)/(bmax.y - bmin.y), 
                               fabs(p.z)/(bmax.z - bmin.z)) ;

       // discern which face is intersected from the largest absolute coordinate 
       // hmm this implicitly assumes a "box" of equal sides, not a "box3"

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
bool csg_intersect_box3(const quad& q0, const float& tt_min, float4& tt, const float3& ray_origin, const float3& ray_direction )
{
   const float3 bmin = make_float3(-q0.f.x/2.f, -q0.f.y/2.f, -q0.f.z/2.f ); 
   const float3 bmax = make_float3( q0.f.x/2.f,  q0.f.y/2.f,  q0.f.z/2.f ); 
   const float3 bcen = make_float3( 0.f, 0.f, 0.f ) ;    

   return _csg_intersect_box( bmin, bmax, bcen , tt_min, tt, ray_origin, ray_direction );   
}
 
static __device__
bool csg_intersect_box(const quad& q0, const float& tt_min, float4& tt, const float3& ray_origin, const float3& ray_direction )
{
   // TODO: get rid of box in favor of box3
   const float hside = q0.f.w ; 
   const float3 bmin = make_float3(q0.f.x - hside, q0.f.y - hside, q0.f.z - hside ); 
   const float3 bmax = make_float3(q0.f.x + hside, q0.f.y + hside, q0.f.z + hside ); 
   const float3 bcen = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

   return _csg_intersect_box( bmin, bmax, bcen , tt_min, tt, ray_origin, ray_direction );   
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
   const unsigned flags = q0.u.w ;

   bool ACAP = flags & SLAB_ACAP ;  
   bool BCAP = flags & SLAB_BCAP ; // b > a by construction

   rtPrintf("## csg_bounds_slab n %7.3f %7.3f %7.3f  a %7.3f b %7.3f flags %u ACAP %d BCAP %d  \n", n.x, n.y, n.z, a, b, flags, ACAP, BCAP );
}

static __device__
bool csg_intersect_slab(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
   const float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    
   const unsigned flags = q0.u.w ;

   const float a = q1.f.x ; 
   const float b = q1.f.y ; 

   bool ACAP = flags & SLAB_ACAP ;  
   bool BCAP = flags & SLAB_BCAP ; // b > a by construction

   float idn = 1.f/dot(ray_direction, n );
   float on = dot(ray_origin, n ); 

   float ta = (a - on)*idn ;
   float tb = (b - on)*idn ;
   
   float t_near = fminf(ta,tb);  // order the intersects 
   float t_far  = fmaxf(ta,tb);

   bool a_near = t_near == ta ;  

   bool near_cap  = a_near ? ACAP : BCAP ; 
   bool far_cap   = a_near ? BCAP : ACAP ;   // a_near -> "b_far"

   float t_cand = t_near > t_min && near_cap ?  t_near : ( t_far > t_min && far_cap  ? t_far : t_min ) ; 

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




   

static __device__
void csg_bounds_cylinder(const quad& q0, const quad& q1, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float3  center = make_float3(q0.f.x, q0.f.y, 0.f ) ;    
    const float   radius = q0.f.w ; 

    const float    z1 = q1.f.x  ; 
    const float    z2 = q1.f.y  ; 

    rtPrintf("## csg_bounds_cylinder center %7.3f %7.3f  radius %7.3f z1 %7.3f z2 %7.3f \n",
          center.x, center.y, radius, z1, z2 );

    const float3 bbmin = make_float3( center.x - radius, center.y - radius, z1 );
    const float3 bbmax = make_float3( center.x + radius, center.y + radius, z2 );

    Aabb tbb(bbmin, bbmax);
    if(tr) transform_bbox( &tbb, tr );  
    aabb->include(tbb);
}



static __device__
void csg_bounds_disc(const quad& q0, const quad& q1, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float3  center = make_float3(q0.f.x, q0.f.y, 0.f ) ;    
    const float   inner  = q0.f.z ; 
    const float   radius = q0.f.w ; 

    const float    z1 = q1.f.x  ; 
    const float    z2 = q1.f.y  ; 

    rtPrintf("## csg_bounds_disc center %7.3f %7.3f  inner %7.3f radius %7.3f  z1 %7.3f z2 %7.3f \n",
          center.x, center.y, inner, radius, z1, z2 );

    const float3 bbmin = make_float3( center.x - radius, center.y - radius, z1 );
    const float3 bbmax = make_float3( center.x + radius, center.y + radius, z2 );

    Aabb tbb(bbmin, bbmax);
    if(tr) transform_bbox( &tbb, tr );  
    aabb->include(tbb);
}


static __device__
bool csg_intersect_disc(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    /*
    RTCD p197

    CSG_DISC was implemented to avoid degeneracy/speckle problems when using CSG_CYLINDER
    to describe very flat cylinders such as Daya Bays ESR mirror surface. 
    Note that the simplicity of disc intersects compared to cylinder has allowed 
    inner radius handling (in param.f.z) for easy annulus definition without using CSG subtraction.

    NB ray-plane intersects are performed with the center disc only at:  z = zc = (z1+z2)/2 
    The t_center obtained is then deltared up and down depending on (z2-z1)/2

    This approach appears to avoid the numerical instability speckling problems encountered 
    with csg_intersect_cylinder when dealing with very flat disc like cylinders. 

    Note that intersects with the edge of the disk are not implemented, if such intersects
    are relevant you need to use CSG_CYLINDER not CSG_DISC.


    For testing see tboolean-esr and tboolean-disc.


                           
                r(t) = O + t n 

                               ^ /         ^ 
                               |/          | d
         ----------------------+-----------|-------------------------------- z2
                              /            |
         - - - - - - - - - - * - - - - - - C- - -  - - - - - - - - - - - - - zc
                            /
         ------------------+------------------------------------------------ z1
                          /|
                         / V
                        /
                       O

          m = O - C



    To work as a CSG sub-object MUST have a different intersect 
    on the other side and normals must be rigidly attached to 
    geometry (must not depend on ray direction)


    Intersect of ray and plane


        r(t) = ray_origin + t * ray_direction

        (r(t) - center).d  = ( m + t * n ).d  = 0    <-- at intersections of ray and plane thru center with normal d 

        t = -m.d / n.d 


    Consider wiggling center up to z2 and down to z1 (in direction of normal d) n.d is unchanged 

        (r(t) - (center+ delta d )).d = 0

        (m - delta d ).d + t * n.d = 0 

        m.d - delta + t* nd = 0 
   
        t =  -(m.d + delta) / n.d              

          = -m.d/n.d  +- delta/n.d


    Intersect is inside disc radius when 

         rsq =   (r(t) - center).(r(t) - center) < radius*radius

         (m + t n).(m + t n)  <  rr

         t*t nn + 2 t nm + mm  <  rr  

         t ( 2 nm + t nn ) + mm   <  rr    

         rsq < rr    checkr(from cylinder) is: rsq - rr 


    Determine whether the t_cand intersect hit after delta-ing 
    is on the upside (normal +Z) or downside (normal -Z) of disc
    from the sign of the below dot product, allowing determination 
    of the rigid outward normal direction.

        r(t) = ray_origin + t * ray_direction

        (r(t_cand) - center).d  = m.d + t_cand n.d     

    */

    const float   inner  = q0.f.z ; 
    const float   radius = q0.f.w ; 
    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ;            // NB z2 > z1 by assertion in npy-/NDisc.cpp
    const float       zc = (z1 + z2)/2.f  ;     // avg
    const float       zdelta = (z2 - z1)/2.f ;  // +ve half difference 

    const float3 center = make_float3( q0.f.x, q0.f.y, zc ); // C: point at middle of disc

    const float3 m = ray_origin - center ;            // m: ray origin in disc frame
    const float3 n = ray_direction ;                  // n: ray direction vector (not normalized)
    const float3 d = make_float3(0.f, 0.f, 1.f );     // d: normal to the disc (normalized)

    float rr = radius*radius ; 
    float ii = inner*inner ; 

    float mm = dot(m, m) ; 
    float nn = dot(n, n) ; 
    float nd = dot(n, d) ;   // >0 : ray direction in same hemi as normal
    float md = dot(m, d) ;
    float mn = dot(m, n) ; 

    float t_center = -md/nd ; 
    float rsq = t_center*(2.f*mn + t_center*nn) + mm  ;   // ( m + tn).(m + tn) 

    float t_delta  = nd < 0.f ? -zdelta/nd : zdelta/nd ;    // <-- pragmatic make t_delta +ve

    float root1 = t_center - t_delta ; 
    float root2 = t_center + t_delta ;   // root2 > root1
 
    float t_cand = ( rsq < rr && rsq > ii ) ? ( root1 > t_min ? root1 : root2 ) : t_min ; 

    float side = md + t_cand*nd ;    

    bool valid_isect = t_cand > t_min ;
    if(valid_isect)
    {        
        isect.x = 0.f ; 
        isect.y = 0.f ; 
        isect.z = side > 0.f ? 1.f : -1.f ; 
        isect.w = t_cand  ; 
 
        return true ; 
    }
    return false ; 
}




/*

#define WITH_CYLINDER_INNER 1

Hmm supporting cylinder with inner radius
directly will add a boatload of cases...


                  /    /        /
                 /    /        /
     +------+   /  . /    +---*--+
     |      |  /   ./     |  /   |
 ----*------*------/------*-/----*------
     |      |/    /.      |/     |
     |      *    / .      *      |
     |     /|   /  .     /|      |
     |    / |  /   .    / |      |
     +---*--+ /    .   /  +------+
        /    /        /
       /    /        /
      /    /        /

Unless could implement by doing internal to the primitive
boolean combination intersects with the full cylinder 
and inner cylinder. Hmm this just duplicates what 
CSG difference or intersect with complement would do though...
so no point.

*/

static __device__
bool csg_intersect_cylinder(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    // ascii art explanation in intersect_ztubs.h

#ifdef WITH_CYLINDER_INNER
    const float inner  = q0.f.z ; 
    const float ii = inner*inner ; 
#endif

    const float   radius = q0.f.w ; 

    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ; 
    const float  sizeZ = z2 - z1 ; 
    const float3 position = make_float3( q0.f.x, q0.f.y, z1 ); // P: point on axis at base of cylinder

    bool PCAP = true ; 
    bool QCAP = true ; 

    const float3 m = ray_origin - position ;          // m: ray origin in cylinder frame
    const float3 n = ray_direction ;                  // n: ray direction vector (not normalized)
    const float3 d = make_float3(0.f, 0.f, sizeZ );   // d: (PQ) cylinder axis vector (not normalized)

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

    float t_cand = t_min ; 

    // axial ray endcap handling 
    if(fabs(a) < 1e-6f)     
    {
        if(c > 0.f) return false ;  // ray starts and ends outside cylinder

        float t_PCAP_AX = -mn/nn  ;
        float t_QCAP_AX = (nd - mn)/nn ;
         
        if(md < 0.f )     // ray origin on P side
        {
            t_cand = PCAP ? t_PCAP_AX : ( QCAP ? t_QCAP_AX : t_min ) ;
        } 
        else if(md > dd )  // ray origin on Q side 
        {
            t_cand = QCAP ? t_QCAP_AX : ( PCAP ? t_PCAP_AX : t_min ) ;
        }
        else              // ray origin inside,   nd > 0 ray along +d towards Q  
        {
            t_cand = nd > 0 ? ( QCAP ? t_QCAP_AX : t_min ) : ( PCAP ? t_PCAP_AX : t_min ) ;  
        }

        unsigned endcap = t_cand == t_PCAP_AX ? CYLINDER_ENDCAP_P : ( t_cand == t_QCAP_AX ? CYLINDER_ENDCAP_Q : 0 ) ;

        bool has_axial_intersect = t_cand > t_min && endcap > 0 ;
        if(has_axial_intersect)
        {
            float sign = endcap == CYLINDER_ENDCAP_P ? -1.f : 1.f ;  
            isect.x = sign*dnorm.x ; 
            isect.y = sign*dnorm.y ; 
            isect.z = sign*dnorm.z ; 
            isect.w = t_cand ;      
        }
        return has_axial_intersect ;
    }   // end-of-axial-ray endcap handling 
    


    if(disc > 0.0f)  // has intersections with the infinite cylinder
    {
        float t_NEAR, t_FAR, sdisc ;   

        robust_quadratic_roots(t_NEAR, t_FAR, disc, sdisc, a, b, c); //  Solving:  a t^2 + 2 b t +  c = 0 

        float t_PCAP = -md/nd ; 
        float t_QCAP = (dd-md)/nd ;   


        float aNEAR = md + t_NEAR*nd ;        // axial coord of near intersection point * sizeZ
        float aFAR  = md + t_FAR*nd ;         // axial coord of far intersection point  * sizeZ

        float3 P1 = ray_origin + t_NEAR*ray_direction ;  
        float3 P2 = ray_origin + t_FAR*ray_direction ;  

        float3 N1  = (P1 - position)/radius  ;   
        float3 N2  = (P2 - position)/radius  ;  

        float checkr = 0.f ; 
        float checkr_PCAP = k + t_PCAP*(2.f*mn + t_PCAP*nn) ; // bracket typo in RTCD book, 2*t*t makes no sense   
        float checkr_QCAP = k + dd - 2.0f*md + t_QCAP*(2.f*(mn-nd)+t_QCAP*nn) ;             


        if( aNEAR > 0.f && aNEAR < dd )  // near intersection inside cylinder z range
        {
            t_cand = t_NEAR ; 
            checkr = -1.f ; 
        } 
        else if( aNEAR < 0.f && PCAP ) //  near intersection outside cylinder z range, on P side
        {
            t_cand =  nd > 0 ? t_PCAP : t_min ;   // nd > 0, ray headed upwards (+z)
            checkr = checkr_PCAP ; 
        } 
        else if( aNEAR > dd && QCAP) //  intersection outside cylinder z range, on Q side
        {
            t_cand = nd < 0 ? t_QCAP : t_min ;  // nd < 0, ray headed downwards (-z) 
            checkr = checkr_QCAP ; 
        }

        // consider looking from P side thru open PCAP towards the QCAP, 
        // the aNEAR will be a long way behind you (due to close to axial ray direction) 
        // hence it will be -ve and thus disqualified as PCAP=false 
        // ... so t_cand will stay at t_min and thus will fall thru 
        // to the 2nd chance intersects 
        

        if( t_cand > t_min && checkr < 0.f )
        {
            isect.w = t_cand ; 
            if( t_cand == t_NEAR )
            {
                isect.x = N1.x ; 
                isect.y = N1.y ; 
                isect.z = 0.f ; 
            } 
            else
            { 
                float sign = t_cand == t_PCAP ? -1.f : 1.f ; 
                isect.x = sign*dnorm.x ; 
                isect.y = sign*dnorm.y ; 
                isect.z = sign*dnorm.z ; 
            }
            return true ; 
        }
       
  
        // resume considing P to Q lookthru, the aFAR >> dd and this time QCAP 
        // is enabled so t_cand = t_QCAP which yields endcap hit so long as checkr_QCAP
        // pans out 
        //
        // 2nd intersect (as RTCD p198 suggests), as the ray can approach 
        // the 2nd endcap from either direction : 
        // 


        if( aFAR > 0.f && aFAR < dd )  // far intersection inside cylinder z range
        {
            t_cand = t_FAR ; 
            checkr = -1.f ; 
        } 
        else if( aFAR < 0.f && PCAP ) //  far intersection outside cylinder z range, on P side (-z)
        {
            t_cand = nd < 0 ? t_PCAP : t_min ;      // sign flip cf RTCD:p198     
            checkr = checkr_PCAP ; 
        } 
        else if( aFAR > dd && QCAP ) //  far intersection outside cylinder z range, on Q side (+z)
        {
            t_cand = nd > 0 ? t_QCAP : t_min  ;    // sign flip cf RTCD:p198
            checkr = checkr_QCAP ;
        }

        if( t_cand > t_min && checkr < 0.f )
        {
            isect.w = t_cand ; 
            if( t_cand == t_FAR )
            {
                isect.x = N2.x ; 
                isect.y = N2.y ; 
                isect.z = 0.f ; 
            } 
            else
            { 
                float sign = t_cand == t_PCAP ? -1.f : 1.f ; 
                isect.x = sign*dnorm.x ; 
                isect.y = sign*dnorm.y ; 
                isect.z = sign*dnorm.z ; 
            } 
            return true ; 
        }

    }  // disc > 0.f

    return false ; 
}

/*
Consider looking thru the open PCAP (-z) end of the cylinder thru to the inside of the QCAP (+z) endcap.

* the infinite cylinder intersects will be far behind (aNEAR < 0) but as


and far ahead (aFAR > dd)
  but only QCAP is enabled


*/




/*


  http://www.cosinekitty.com/raytrace/chapter13_torus.html

    (xx + yy + zz + RR - rr )^2 = 4RR(xx + yy)


*/





static __device__
void csg_bounds_torus(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float rminor = q0.f.z ;     
    const float rmajor = q0.f.w ;    
    const float rsum = rminor + rmajor ;  

    rtPrintf("## csg_bounds_torus rmajor %f rminor %f rsum %f  \n", rmajor, rminor, rsum );

    float3 mn = make_float3( -rsum, -rsum,  -rminor );
    float3 mx = make_float3(  rsum,  rsum,   rminor );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}




static __device__
float cubic_delta2( const float a, const float b, const float c, const float d)
{
    // Neumark p5  :   a x^3 + b x^2 + c x + d = 0 
    //
    float tmp = 27.f*a*a*d + 2.f*b*b*b - 9.f*a*b*c ; 
    float disc = b*b - 3.f*a*c ; 
    float delta2 = tmp*tmp / (2.f*disc*disc*disc) ; 
    return delta2 ;
}



#define TORUS_DEBUG 1

static __device__
bool csg_intersect_torus(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    const float r = q0.f.z ; 
    const float R = q0.f.w ;  // R > r by assertion, so torus has a hole

    const float rr = r*r ; 
    const float RR = R*R ; 
    const float RR4 = R*R*4.0f ; 

    const float& ox = ray_origin.x ; 
    const float& oy = ray_origin.y ; 

    const float& sx = ray_direction.x ;
    const float& sy = ray_direction.y ;

    // try using ray-circle discriminant to skip SolveQuartic
    // hmm for this to work must rotate line about z-axis to bring into
    // the testing plane ... that means translate the ray to origin rotating 
    // then untranslating...
    // ... or  
    // alternatively consider cone sphere intersect ?
    /*
    const float& oz = ray_origin.z ; 
    const float& sz = ray_direction.z ;

    const float rc2 = sx*sx + sz*sz ; 
    const float rc1_l = -2.f*(R*sx - ox*sx - oz*sz) ;
    const float rc1_r = -2.f*(-R*sx - ox*sx - oz*sz) ;
    const float rc0 = RR - 2.f*R*ox - rr + ox*ox + oz*oz ;
    const float rc_discr_l = rc1_l*rc1_l - 4.f*rc2*rc0 ; 
    const float rc_discr_r = rc1_r*rc1_r - 4.f*rc2*rc0 ; 
    if(rc_discr_l < 0.f && rc_discr_r < 0.f) return false ; 
    */


    // following cosinekitty nomenclature, sympy verified in torus.py
    const float G = RR4*(sx*sx+sy*sy) ; 
    const float H = 2.f*RR4*(ox*sx+oy*sy) ; 
    const float I = RR4*(ox*ox+oy*oy) ;
    const float J = dot(ray_direction, ray_direction) ;
    const float K = 2.f*dot(ray_origin, ray_direction) ;
    const float L = dot(ray_origin, ray_origin) + RR - rr ; 


    float q[5]; 
    q[4] = J*J ; 
    q[3] = 2.f*J*K ; 
    q[2] = 2.f*J*L + K*K - G ;
    q[1] = 2.f*K*L - H ;
    q[0] = L*L - I ;

    float qn[4] ; 
    qn[3] = q[3]/q[4] ;
    qn[2] = q[2]/q[4] ;
    qn[1] = q[1]/q[4] ;
    qn[0] = q[0]/q[4] ;


#ifdef TORUS_DEBUG 
    const float a = qn[3] ; 
    const float b = qn[2] ; 
    const float c = qn[1] ; 
    const float d = qn[0] ; 

    // coeff of depressed quartic
    const float e     = b - 3.f * a * a / 8.f;
    const float f     = c + a * a * a / 8.f - 0.5f * a * b;
    const float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;

    // switch x -> -x and multiplying by -1 , will flip even x^2, x^0 coeff signs    

    float neumark[3] ;  // coeff of resolvent cubic
    neumark[2] = 2.f*e ; 
    neumark[1] = e*e - 4.f*g ;
    neumark[0] = -f*f ;

    float neumark_delta2 = cubic_delta2( 1.f, neumark[2], neumark[1], neumark[0] ); 
    bool double_root = fabsf( neumark_delta2 - 1.0f ) < 1.e-3f ; 

#endif

    float roots[4] ;   

   // unsigned msk = SOLVE_VECGEOM  ;  // worst for artifcacting 
   // unsigned msk = SOLVE_UNOBFUSCATED  ;  // reduced but still obvious
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD ;  // getting some in-out wierdness
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD | SOLVE_ROBUSTCUBIC_0 ;
   unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ;  // _0 ok
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_1 ;  // in-out wierdness
 
    

    //int num_roots = SolveQuartic( qn[3],qn[2],qn[1],qn[0], roots, msk ); 
    int num_roots = SolveQuarticPureNeumark( qn[3],qn[2],qn[1],qn[0], roots, msk ); 


    float4 cand = make_float4(RT_DEFAULT_MAX) ;  
    int num_cand = 0 ;  
    if(num_roots > 0)
    {
        for(int i=0 ; i < num_roots ; i++)
        {
            if(roots[i] > t_min ) setByIndex(cand, num_cand++, roots[i]) ; 
        }   
    }
    
    float t_cand = num_cand > 0 ? fminf(cand) : t_min ;   // smallest root bigger than t_min

    const float3 p = ray_origin + t_cand*ray_direction ; 


    const float2 qrz = make_float2(length(make_float2(p)) - R, p.z) ;  // (mid-circle-sdf, z)
    const float  qsd = length(qrz) - r ;   // signed dist to torus
    bool valid_qsd = fabsf(qsd) < 1e-3f ; 

    // Can easily cut away fake intersects assumed caused by numerical problems 
    // for some coeffient combinations using sdf... 
    // but how to handle failed intersects ?
    //
    // Inverting !valid_qsd to see just the fakes
    // suggests interior missing intersects and
    // exterior fake intersects arise from same cause.
    //
    // So identifying cause of exterior fakes (eg some coeff going to zero)
    // may potentially also fix interior missings.
         



#ifndef TORUS_DEBUG 
    bool valid_isect = valid_qsd && t_cand > t_min ;
#else
    const float residual = (((q[4]*t_cand + q[3])*t_cand + q[2])*t_cand + q[1])*t_cand + q[0] ; 

    //bool valid_isect = t_cand > t_min && !double_root ;   // double roots not source of artifacts
    //bool valid_isect = t_cand > t_min  ;
    //bool valid_isect = t_cand > t_min && qsd < -0.1f   ;
    //bool valid_isect = t_cand > t_min && fabsf(neumark[0]) < 0.1f && fabsf(residual) > 0.1f  ; // ring artifact
    //bool valid_isect = t_cand > t_min && fabsf(neumark[0]) > 0.1f && fabsf(residual) > 0.1f  ; //  nowt so ring is mostly from small neumark[0] ? 

    //bool valid_isect = t_cand > t_min && fabsf(residual) < 0.1f  ;  // hailine ring crack
    //bool valid_isect = t_cand > t_min && fabsf(neumark[0]) > 0.001f ;  // > 0.001f entirely avoids artifacting (following ROBUST fixes) but chops ring band out of torus
    //bool valid_isect = t_cand > t_min && fabsf(neumark[0]) > 0.0001f ;  // > 0.0001f artifact ring visible and chops thin ring band out of torus
    bool valid_isect = t_cand > t_min ;
    //bool valid_isect = valid_qsd && t_cand > t_min ;

    if(valid_isect) 
    {

        const float A = q[4] ; 
        const float B = q[3] ; 
        const float C = q[2] ; 
        const float D = q[1] ; 
        const float E = q[0] ; 

        // Neumark p18 : Resolvent cubic has one zero root 
        //  Hurwitz-Roth discriminant for oscillatory stability...
        //  leads to two equal and opposite real roots when zero

        const float HRD = B*C*D - B*B*E - A*D*D ;   // huh expecte to be small at problem points, but not so
    
        // Hmm perhaps pick resolvent cubic that avoid numerical problems ?
        
        rtPrintf(
                 "torus"
                 " num_roots %d "
                 " num_cand %d "
                 " t_cand %10.3g "
                 " roots ( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 " residual %10.4f "
                 " qsd %10.4f "
                 "\n"
                 ,
                 num_roots
                 ,
                 num_cand
                 ,
                 t_cand
                 ,
                 roots[0], roots[1], roots[2], roots[3]
                 ,
                 residual
                 ,
                 qsd
               );
    }

/*    


                 " neumark_delta2 : %10.3g "
                 ,
                 neumark_delta2



                 " qn( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 " efg( %10.3g, %10.3g, %10.3g )"
                 " neumark( %10.3g, %10.3g, %10.3g )"

                 qn[3],qn[2],qn[1],qn[0]
                 ,
                 e,f,g
                 ,
                 neumark[2],neumark[1],neumark[0]





                 " HRD %10.4f "
                 HRD
 

                 " R %10.4f r %10.4f "
                  R, r,                      // 1, 0.5


                 " qsd %10.4f "
                 qsd, 
                 qsd    14.5927  

                 " dir (%10.4g %10.4g %10.4g) "
                 ray_direction.x, ray_direction.y, ray_direction.z,

                 dir ( -0.001487 -0.0009288  -0.009845)    ## length of this sq uncomfortably small, following from non-norm for scaling 
                 dir (  -0.07532     0.1403    -0.9872)    ## removing scaling in tboolean-torus makes more healthy


                 " ori (%10.4g %10.4g %10.4g) "
                 ray_origin.x, ray_origin.y, ray_origin.z,

                 ori (    -0.103      2.134     -6.638)     ## as expect


                 " GHIJKL( %10.3g, %10.3g, %10.3g, %10.3g, %10.3g, %10.3g)"     
                 G,H,I,J,K,L

                 GHIJKL(   1.23e-05,    -0.0152,        7.2,     0.0001,     -0.118,       36.1)   ## with scaling of 100
                 GHIJKL(      0.745,      -5.72,       16.9,          1,      -7.62,       16.8)   ## without scaling J=1 norm ray dir
    

                 " q( %10.3g, %10.3g, %10.3g, %10.3g, %10.3g)"    
                 q[4],q[3],q[2],q[1],q[0]     


                 q(      1e-08,  -2.09e-05,     0.0308,      -20.8,   9.79e+03)    ##   v.small q[4] from J**2 from unnorm ray dir
                 qsd   108.6718   q(          1,      -15.2,       90.6,       -249,        263)   ## more reasonable q when avoid scaling


                 " qn( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 qn[3],qn[2],qn[1],qn[0],


                 qsd    52.0278  q(          1,      -14.8,       86.5,       -233,        243) 
                 qn(      -14.8,       86.5,       -233,        243)



*/



#endif


    if(valid_isect)
    {        
        const float alpha = 1.f - (R/sqrt(p.x*p.x+p.y*p.y)) ;   // see cosinekitty 
        // how to normalize by construction ?
        const float3 n = normalize(make_float3(alpha*p.x, alpha*p.y, p.z ));
        isect.x = n.x ;  
        isect.y = n.y ;  
        isect.z = n.z ;  
        isect.w = t_cand ; 
    }
    return valid_isect ; 
}


/*

   glm::vec2 q( glm::length(glm::vec2(p)) - rmajor() , p.z );
    float sd = glm::length(q) - rminor() ;


*/

