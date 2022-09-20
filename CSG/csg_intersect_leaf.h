#pragma once
/**
csg_intersect_leaf.h : distance_leaf and intersect_leaf functions
===================================================================

Thus header needs to be included before csg_intersect_node.h which needs to be included before csg_intersect_tree.h 

distance_leaf_sphere 
intersect_leaf_sphere
    CSG_SPHERE, robust_quadratic_roots

distance_leaf_zsphere 
intersect_leaf_zsphere
    CSG_ZSPHERE, robust_quadratic_roots 

distance_leaf_convexpolyhedron
intersect_leaf_convexpolyhedron
    CSG_CONVEXPOLYHEDRON, plane intersections

MISSING : distance_leaf_cone
intersect_leaf_cone
    CSG_CONE, newcone with robust_quadratic_roots, oldcone without

MISSING : distance_leaf_hyperboloid
intersect_leaf_hyperboloid
    CSG_HYPERBOLOID, robust_quadratic_roots 

distance_leaf_box3
intersect_leaf_box3
    CSG_BOX3, plane intersections

distance_leaf_plane
intersect_leaf_plane
    CSG_PLANE

MISSING : distance_leaf_phicut 
intersect_leaf_phicut
    CSG_PHICUT 

distance_leaf_slab
intersect_leaf_slab
    CSG_SLAB

distance_leaf_cylinder
intersect_leaf_cylinder
    CSG_CYLINDER, robust_quadratic_roots_disqualifying 

MISSING : distance_leaf_infcylinder
intersect_leaf_infcylinder
    CSG_INFCYLINDER, robust_quadratic_roots

MISSING : distance_leaf_disc
intersect_leaf_disc
    CSG_DISC, disc still using the pseudo-general flop-heavy approach similar to oldcylinder
  
    * TODO: adopt less-flops approach like newcylinder
    * (NOT URGENT AS disc NOT CURRENTLY VERY RELEVANT IN ACTIVE GEOMETRIES) 


distance_leaf
intersect_leaf



Bringing over functions from  ~/opticks/optixrap/cu/csg_intersect_primitive.h

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LEAF_FUNC __forceinline__ __device__
#else
#    define LEAF_FUNC inline
#endif


#define RT_DEFAULT_MAX 1.e27f

#if defined(__CUDACC__)
#include "math_constants.h"
#else

union uif_t 
{
    unsigned u ; 
    int i ; 
    float f ; 
};

LEAF_FUNC
float __int_as_float(int i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}

#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_PI_F             3.141592654f

#endif


#include "OpticksCSG.h"
#include "squad.h"

#include "CSGNode.h"
#include "CSGPrim.h"

#include "csg_robust_quadratic_roots.h"
#include "csg_classify.h"

#ifdef DEBUG_RECORD
#include <csignal>
#endif

#ifdef DEBUG_CYLINDER
#include "CSGDebug_Cylinder.hh"
#endif


LEAF_FUNC
float distance_leaf_sphere(const float3& pos, const quad& q0 )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;
    float3 p = pos - center;
    float sd = length(p) - radius ; 
    return sd ; 
}


LEAF_FUNC
bool intersect_leaf_sphere(float4& isect, const quad& q0, const float& t_min, const float3& ray_origin, const float3& ray_direction )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

    float3 O = ray_origin - center;
    float3 D = ray_direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);

#ifdef CATASTROPHIC_SUBTRACTION_ROOTS
    float disc = b*b-d*c;   // when d*c small,  sdisc ~ b => catastrophic precision loss in root2 = (-b + sdisc)/d
    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // repeated root for sdisc 0.f
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always
#else
    float root1, root2, disc, sdisc ;   
    robust_quadratic_roots(root1, root2, disc, sdisc, d, b, c ) ; //  Solving:  d t^2 + 2 b t +  c = 0    root2 > root1 
#endif

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ;

    bool valid_isect = t_cand > t_min ;
    if(valid_isect)
    {
        isect.x = (O.x + t_cand*D.x)/radius ;   // normalized by construction
        isect.y = (O.y + t_cand*D.y)/radius ;
        isect.z = (O.z + t_cand*D.z)/radius ;
        isect.w = t_cand ;
    }

#ifdef DEBUG
    //printf("//intersect_leaf_sphere valid_isect %d  isect ( %10.4f %10.4f %10.4f %10.4f)  \n", valid_isect, isect.x, isect.y, isect.z, isect.w ); 
    printf("//intersect_leaf_sphere valid %d radius %10.4f center (%10.4f, %10.4f, %10.4f) ray_ori (%10.4f, %10.4f, %10.4f)  \n", 
       valid_isect,  radius, center.x, center.y, center.z, ray_origin.x, ray_origin.y, ray_origin.z  );  
#endif


    return valid_isect ;
}




LEAF_FUNC
float distance_leaf_zsphere(const float3& pos, const quad& q0, const quad& q1 )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;
    const float2 zdelta = make_float2(q1.f);
    const float z2 = center.z + zdelta.y ; 
    const float z1 = center.z + zdelta.x ;    

    float3 p = pos - center;
    float sd_sphere = length(p) - radius ; 
    float sd_capslab = fmaxf( pos.z - z2 , z1 - pos.z ); 

    float sd = fmaxf( sd_capslab, sd_sphere );    // CSG intersect
    return sd ; 
}

/**
intersect_leaf_zsphere
------------------------

HMM: rays that look destined to land near to "apex" have a rare (order 1 in 300k) 
problem of missing the zsphere.  This is probably arising from the upper cap 
implementation acting effectively like cutting a pinhole at the apex. 

When there is no upper cap perhaps can avoid the problem by setting zmax to beyond the 
apex ? Or could have a different imp for zsphere with lower cap but no upper cap. 

Note that zsphere with no upper cap is used a lot for PMTs so a simpler imp
for zsphere without upper cut does make sense.  

NB "z2sph <= zmax" changed from "z2sph < zmax" Aug 29, 2022

The old inequality caused rare unexpected MISS for rays that would
have been expected to intersect close to the apex of the zsphere  

See : notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst

**/


LEAF_FUNC
bool intersect_leaf_zsphere(float4& isect, const quad& q0, const quad& q1, const float& t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float3 center = make_float3(q0.f);
    float3 O = ray_origin - center;  
    float3 D = ray_direction;
    const float radius = q0.f.w;

    float b = dot(O, D);               // t of closest approach to sphere center
    float c = dot(O, O)-radius*radius; // < 0. indicates ray_origin inside sphere

#ifdef DEBUG_RECORD
    printf("//[intersect_leaf_zsphere radius %10.4f b %10.4f c %10.4f \n", radius, b, c); 
#endif

    if( c > 0.f && b > 0.f ) return false ;    
    // Cannot intersect when ray origin outside sphere and direction away from sphere.
    // Whether early exit speeds things up is another question ... 

    const float2 zdelta = make_float2(q1.f);
    const float zmax = center.z + zdelta.y ;   // + 0.1f artificial increase zmax to test apex bug 
    const float zmin = center.z + zdelta.x ;    

#ifdef DEBUG_RECORD
    bool with_upper_cut = zmax < radius ; 
    bool with_lower_cut = zmin > -radius ; 
    printf("// intersect_leaf_zsphere radius %10.4f zmax %10.4f zmin %10.4f  with_upper_cut %d with_lower_cut %d  \n", radius, zmax, zmin, with_upper_cut, with_lower_cut ); 
#endif


    float d = dot(D, D);               // NB NOT assuming normalized ray_direction

    float t1sph, t2sph, disc, sdisc ;    
    robust_quadratic_roots(t1sph, t2sph, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

    float z1sph = ray_origin.z + t1sph*ray_direction.z ;  // sphere z intersects
    float z2sph = ray_origin.z + t2sph*ray_direction.z ; 

#ifdef DEBUG_RECORD
    printf("// intersect_leaf_zsphere t1sph %10.4f t2sph %10.4f sdisc %10.4f \n", t1sph, t2sph, sdisc ); 
    printf("// intersect_leaf_zsphere z1sph %10.4f z2sph %10.4f zmax %10.4f zmin %10.4f sdisc %10.4f \n", z1sph, z2sph, zmax, zmin, sdisc ); 
#endif

    float idz = 1.f/ray_direction.z ; 
    float t_QCAP = (zmax - ray_origin.z)*idz ;   // upper cap intersects
    float t_PCAP = (zmin - ray_origin.z)*idz ;   // lower cap intersect 


    float t1cap = fminf( t_QCAP, t_PCAP ) ;      // order cap intersects along the ray 
    float t2cap = fmaxf( t_QCAP, t_PCAP ) ;      // t2cap > t1cap 

#ifdef DEBUG_RECORD
    bool t1cap_disqualify = t1cap < t1sph || t1cap > t2sph ; 
    bool t2cap_disqualify = t2cap < t1sph || t2cap > t2sph ;  
    printf("//intersect_leaf_zsphere t1sph %7.3f t2sph %7.3f t_QCAP %7.3f t_PCAP %7.3f t1cap %7.3f t2cap %7.3f  \n", t1sph, t2sph, t_QCAP, t_PCAP, t1cap, t2cap ); 
    printf("//intersect_leaf_zsphere  t1cap_disqualify %d t2cap_disqualify %d \n", t1cap_disqualify, t2cap_disqualify  ); 
#endif

    // disqualify plane intersects outside sphere t range
    if(t1cap < t1sph || t1cap > t2sph) t1cap = t_min ; 
    if(t2cap < t1sph || t2cap > t2sph) t2cap = t_min ; 

    // hmm somehow is seems unclean to have to use both z and t language

    float t_cand = t_min ; 
    if(sdisc > 0.f)
    {

#ifdef DEBUG_RECORD
        //std::raise(SIGINT); 
#endif

        if(      t1sph > t_min && z1sph > zmin && z1sph <= zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
        else if( t1cap > t_min )                                   t_cand = t1cap ;  // t1cap qualifies -> t1cap 
        else if( t2cap > t_min )                                   t_cand = t2cap ;  // t2cap qualifies -> t2cap
        else if( t2sph > t_min && z2sph > zmin && z2sph <= zmax)   t_cand = t2sph ;  // t2sph qualifies and t2cap disabled or disqialified -> t2sph
    }

    bool valid_isect = t_cand > t_min ;
#ifdef DEBUG_RECORD
    printf("//intersect_leaf_zsphere valid_isect %d t_min %7.3f t1sph %7.3f t1cap %7.3f t2cap %7.3f t2sph %7.3f t_cand %7.3f \n", valid_isect, t_min, t1sph, t1cap, t2cap, t2sph, t_cand ); 
#endif

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

#ifdef DEBUG_RECORD
    printf("//]intersect_leaf_zsphere valid_isect %d \n", valid_isect ); 
#endif
    return valid_isect ;
}


LEAF_FUNC
float distance_leaf_convexpolyhedron( const float3& pos, const CSGNode* node, const float4* plan )
{
    unsigned planeIdx = node->planeIdx() ; 
    unsigned planeNum = node->planeNum() ; 
    float sd = 0.f ; 
    for(unsigned i=0 ; i < planeNum ; i++) 
    {    
        const float4& plane = plan[planeIdx+i];   
        float d = plane.w ;
        float3 n = make_float3(plane);
        float sd_plane = dot(pos, n) - d ; 
        sd = i == 0 ? sd_plane : fmaxf( sd, sd_plane ); 
    }
    return sd ; 
}


LEAF_FUNC
bool intersect_leaf_convexpolyhedron( float4& isect, const CSGNode* node, const float4* plan, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float t0 = -CUDART_INF_F ; 
    float t1 =  CUDART_INF_F ; 

    float3 t0_normal = make_float3(0.f);
    float3 t1_normal = make_float3(0.f);

    unsigned planeIdx = node->planeIdx() ; 
    unsigned planeNum = node->planeNum() ; 

    for(unsigned i=0 ; i < planeNum ; i++) 
    {    
        const float4& plane = plan[planeIdx+i];   
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

        if(parallel_inside) continue ;       // continue to next plane 
        if(parallel_outside) return false ;  // <-- without early exit, this still works due to infinity handling 

        //    NB ray parallel to plane and outside halfspace 
        //         ->  t_cand = -inf 
        //                 nd = 0.f 
        //                t1 -> -inf  

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




LEAF_FUNC
float z_apex_cone( const quad& q0 )
{
    float r1 = q0.f.x ; 
    float z1 = q0.f.y ; 
    float r2 = q0.f.z ; 
    float z2 = q0.f.w ;   // z2 > z1
    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex
    return z0 ; 
}



#include "csg_intersect_leaf_newcone.h"
#include "csg_intersect_leaf_oldcone.h"


/**
intersect_leaf_hyperboloid
-----------------------------

* http://mathworld.wolfram.com/One-SheetedHyperboloid.html

::

      x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )
      x^2 + y^2 - (r0^2/zf^2) * z^2 - r0^2  =  0 
      x^2 + y^2 + A * z^2 + B   =  0 

      grad( x^2 + y^2 + A * z^2 + B ) =  [2 x, 2 y, A*2z ] 


     (ox+t sx)^2 + (oy + t sy)^2 + A (oz+ t sz)^2 + B = 0 

      t^2 ( sxsx + sysy + A szsz ) + 2*t ( oxsx + oysy + A * ozsz ) +  (oxox + oyoy + A * ozoz + B ) = 0 

**/

LEAF_FUNC
bool intersect_leaf_hyperboloid(float4& isect, const quad& q0, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float zero(0.f); 
    const float one(1.f); 

    const float r0 = q0.f.x ;  // waist (z=0) radius 
    const float zf = q0.f.y ;  // at z=zf radius grows to  sqrt(2)*r0 
    const float z1 = q0.f.z ;  // z1 < z2 by assertion  
    const float z2 = q0.f.w ;  

    const float rr0 = r0*r0 ;
    const float z1s = z1/zf ; 
    const float z2s = z2/zf ; 
    const float rr1 = rr0 * ( z1s*z1s + one ) ; // radii squared at z=z1, z=z2
    const float rr2 = rr0 * ( z2s*z2s + one ) ;

    const float A = -rr0/(zf*zf) ;
    const float B = -rr0 ;  

    const float& sx = ray_direction.x ; 
    const float& sy = ray_direction.y ; 
    const float& sz = ray_direction.z ;

    const float& ox = ray_origin.x ; 
    const float& oy = ray_origin.y ; 
    const float& oz = ray_origin.z ;

    const float d = sx*sx + sy*sy + A*sz*sz ; 
    const float b = ox*sx + oy*sy + A*oz*sz ; 
    const float c = ox*ox + oy*oy + A*oz*oz + B ; 
    
    float t1hyp, t2hyp, disc, sdisc ;   
    robust_quadratic_roots(t1hyp, t2hyp, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

    const float h1z = oz + t1hyp*sz ;  // hyp intersect z positions
    const float h2z = oz + t2hyp*sz ; 

    //  z = oz+t*sz -> t = (z - oz)/sz 
    float osz = one/sz ; 
    float t2cap = (z2 - oz)*osz ;   // cap plane intersects
    float t1cap = (z1 - oz)*osz ;

    const float3 c1 = ray_origin + t1cap*ray_direction ; 
    const float3 c2 = ray_origin + t2cap*ray_direction ; 

    float crr1 = c1.x*c1.x + c1.y*c1.y ;   // radii squared at cap plane intersects
    float crr2 = c2.x*c2.x + c2.y*c2.y ; 

    // NB must disqualify t < t_min at "front" and "back" 
    // as this potentially picks between hyp intersects eg whilst near(t_min) scanning  

    const float4 t_cand_ = make_float4(   // restrict radii of cap intersects and z of hyp intersects
                                          t1hyp > t_min && disc > zero && h1z > z1 && h1z < z2 ? t1hyp : RT_DEFAULT_MAX ,
                                          t2hyp > t_min && disc > zero && h2z > z1 && h2z < z2 ? t2hyp : RT_DEFAULT_MAX ,
                                          t2cap > t_min && crr2 < rr2                          ? t2cap : RT_DEFAULT_MAX ,
                                          t1cap > t_min && crr1 < rr1                          ? t1cap : RT_DEFAULT_MAX 
                                      ) ;

    float t_cand = fminf( t_cand_ );  

    bool valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
    if(valid_isect)
    {        
        isect.w = t_cand ; 
        if( t_cand == t1hyp || t_cand == t2hyp )
        {
            const float3 p = ray_origin + t_cand*ray_direction ; 
            float3 n = normalize(make_float3( p.x,  p.y,  A*p.z )) ;   // grad(level-eqn) 
            isect.x = n.x ; 
            isect.y = n.y ; 
            isect.z = n.z ;      
        }
        else
        {
            isect.x = zero ; 
            isect.y = zero ; 
            isect.z = t_cand == t1cap ? -one : one ;  
        }
    }
    return valid_isect ; 
}


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



/**
distance_leaf_cylinder
------------------------

* https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

Capped Cylinder - exact

float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r); 
  // dont follow  would expect h <-> r with radius to be on the first dimension and height on second
     
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}


                      p
                      +
                      | 
                      |
                      | 
  - - - +---r----+----+---+ - - - - - -
        |        :        |
        h        :        +------+ p    
        |        :        |
        |        :        |
        +--------+--------+
        |        :        |
        |        :        |
        |        :        |
        |        :        |
  - - - +--------+--------+ - - - - - - - 





The SDF rules for CSG combinations::

    CSG union(l,r)     ->  min(l,r)
    CSG intersect(l,r) ->  max(l,r)
    CSG difference(l,r) -> max(l,-r)    [aka subtraction, corresponds to intersecting with a complemented rhs]


**/


LEAF_FUNC
float distance_leaf_cylinder( const float3& pos, const quad& q0, const quad& q1 )
{
    const float   radius = q0.f.w ; 
    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ;   // z2 > z1 

    float sd_capslab = fmaxf( pos.z - z2 , z1 - pos.z ); 
    float sd_infcyl = sqrtf( pos.x*pos.x + pos.y*pos.y ) - radius ;  
    float sd = fmaxf( sd_capslab, sd_infcyl ); 

#ifdef DEBUG
    printf("//distance_leaf_cylinder sd %10.4f \n", sd ); 
#endif
    return sd ; 
}


#include "csg_intersect_leaf_oldcylinder.h"

/**
intersect_leaf_cylinder : a much simpler approach than intersect_leaf_oldcylinder
-------------------------------------------------------------------------------------------

The two cylinder imps were compared with tests/CSGIntersectComparisonTest.cc.
Surface distance comparisons show the new imp is more precise and does
not suffer from near-axial spurious intersects beyond the ends.  

intersect_leaf_cylinder

   * simple as possible approach, minimize the flops
   * axial special case removed, might need to put back if find some motivation to do that

intersect_leaf_oldcylinder

   * pseudo-general approach, based on implementation from book RTCD  
   * had axial special case bolted on for unrecorded reason, some glitch presumably 


There are four possible t

* 2 from curved sheet, obtained from solving quadratic, that must be within z1 z2 range
* 2 from endcaps that must be within r2 range  

Finding the intersect means finding the smallest t from the four that exceeds t_min  

Current approach keeps changing t_cand, could instead collect all four potential t 
into a float4 and then pick from that ? 

**/

LEAF_FUNC
bool intersect_leaf_cylinder( float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float& r  = q0.f.w ; 
    const float& z1 = q1.f.x  ; 
    const float& z2 = q1.f.y  ; 
    const float& ox = ray_origin.x ; 
    const float& oy = ray_origin.y ; 
    const float& oz = ray_origin.z ; 
    const float& vx = ray_direction.x ; 
    const float& vy = ray_direction.y ; 
    const float& vz = ray_direction.z ; 

    const float r2 = r*r ; 
    const float a = vx*vx + vy*vy ;     // see CSG/sympy_cylinder.py 
    const float b = ox*vx + oy*vy ; 
    const float c = ox*ox + oy*oy - r2 ; 

    float t_near, t_far, disc, sdisc ;   
    robust_quadratic_roots_disqualifying(t_min, t_near, t_far, disc, sdisc, a, b, c); //  Solving:  a t^2 + 2 b t +  c = 0 
    float z_near = oz+t_near*vz ; 
    float z_far  = oz+t_far*vz ; 

    const float t_z1cap = (z1 - oz)/vz ; 
    const float r2_z1cap = (ox+t_z1cap*vx)*(ox+t_z1cap*vx) + (oy+t_z1cap*vy)*(oy+t_z1cap*vy) ;  

    const float t_z2cap = (z2 - oz)/vz ;  
    const float r2_z2cap = (ox+t_z2cap*vx)*(ox+t_z2cap*vx) + (oy+t_z2cap*vy)*(oy+t_z2cap*vy) ;  

#ifdef DEBUG
    //printf("// t_z1cap %10.4f r2_z1cap %10.4f \n", t_z1cap, r2_z1cap ); 
    //printf("// t_z2cap %10.4f r2_z2cap %10.4f \n", t_z2cap, r2_z2cap ); 
#endif

    float t_cand = CUDART_INF_F ;
    if( t_near  > t_min && z_near   > z1 && z_near < z2 && t_near  < t_cand ) t_cand = t_near ; 
    if( t_far   > t_min && z_far    > z1 && z_far  < z2 && t_far   < t_cand ) t_cand = t_far ; 
    if( t_z1cap > t_min && r2_z1cap <= r2               && t_z1cap < t_cand ) t_cand = t_z1cap ; 
    if( t_z2cap > t_min && r2_z2cap <= r2               && t_z2cap < t_cand ) t_cand = t_z2cap ; 

    bool valid_intersect = t_cand > t_min && t_cand < CUDART_INF_F ; 
    if(valid_intersect)
    {
        bool sheet = ( t_cand == t_near || t_cand == t_far ) ; 
        isect.x = sheet ? (ox + t_cand*vx)/r : 0.f ; 
        isect.y = sheet ? (oy + t_cand*vy)/r : 0.f ; 
        isect.z = sheet ? 0.f : ( t_cand == t_z1cap ? -1.f : 1.f) ; 
        isect.w = t_cand ;      
    }
    return valid_intersect ; 
}



/**
intersect_leaf_infcylinder
----------------------------------

Use standard Z-axial cylinder orientation to see how much it simplifies::

    x^2 + y^2 = r^2

    (Ox + t Dx)^2 + (Oy + t Dy)^2 = r^2 

    Ox ^2 + t^2 Dx^2 + 2 t Ox Dx   
    Oy ^2 + t^2 Dy^2 + 2 t Oy Dy


    t^2 (Dx^2 + Dy^2) + 2 t ( OxDx + Oy Dy ) + Ox^2 + Oy^2 - r^2  = 0     

Contrast this eqn with that on RTCD p195 "bk-;bk-rtcd 195"  : its a natural simplification.
Instead of dotting all components and subtracting the axial part can just directly 
dot the non-axial x and y thanks to the fixed orientation.

**/

LEAF_FUNC
bool intersect_leaf_infcylinder( float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float r = q0.f.w ; 

    const float3& O = ray_origin ;    
    const float3& D = ray_direction ;    

    float a = D.x*D.x + D.y*D.y ; 
    float b = O.x*D.x + O.y*D.y ; 
    float c = O.x*O.x + O.y*O.y - r*r ;  

    float disc = b*b-a*c;

    if(disc > 0.0f)  // has intersections with the infinite cylinder
    {
        float t_NEAR, t_FAR, sdisc ;   

        robust_quadratic_roots(t_NEAR, t_FAR, disc, sdisc, a, b, c); //  Solving:  a t^2 + 2 b t +  c = 0 

        float t_cand = sdisc > 0.f ? ( t_NEAR > t_min ? t_NEAR : t_FAR ) : t_min ;

        bool valid_isect = t_cand > t_min ; 

        if( valid_isect  )
        {
            isect.x = (O.x + t_cand*D.x)/r ;   // normalized by construction
            isect.y = (O.y + t_cand*D.y)/r ;
            isect.z = 0.f ;
            isect.w = t_cand ; 
            return true ; 
        }
    }
    return false ; 
}
 


/**
intersect_leaf_disc
---------------------

RTCD p197  (Real Time Collision Detection)

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


For testing see tboolean-esr and tboolean-disc.::

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


Intersect of ray and plane::

    r(t) = ray_origin + t * ray_direction

    (r(t) - center).d  = ( m + t * n ).d  = 0    <-- at intersections of ray and plane thru center with normal d 

    t = -m.d / n.d 

Consider wiggling center up to z2 and down to z1 (in direction of normal d) n.d is unchanged::

    (r(t) - (center+ delta d )).d = 0

    (m - delta d ).d + t * n.d = 0 

    m.d - delta + t* nd = 0 

    t =  -(m.d + delta) / n.d              

      = -m.d/n.d  +- delta/n.d


Intersect is inside disc radius when::

    rsq =   (r(t) - center).(r(t) - center) < radius*radius

    (m + t n).(m + t n)  <  rr

    t*t nn + 2 t nm + mm  <  rr  

    t ( 2 nm + t nn ) + mm   <  rr    

    rsq < rr    checkr(from cylinder) is: rsq - rr 


Determine whether the t_cand intersect hit after delta-ing 
is on the upside (normal +Z) or downside (normal -Z) of disc
from the sign of the below dot product, allowing determination 
of the rigid outward normal direction.::

    r(t) = ray_origin + t * ray_direction

    (r(t_cand) - center).d  = m.d + t_cand n.d     

**/

LEAF_FUNC
bool intersect_leaf_disc(float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float   inner  = q0.f.z ; 
    const float   radius = q0.f.w ; 
    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ;            // NB z2 > z1 by assertion in npy-/NDisc.cpp
    const float       zc = (z1 + z2)/2.f  ;     // avg
    const float       zdelta = (z2 - z1)/2.f ;  // +ve half difference 

    const float3 center = make_float3( q0.f.x, q0.f.y, zc ); // C: point at middle of disc

#ifdef DEBUG
    printf("//intersect_leaf_disc (%10.4f, %10.4f, %10.4f) \n", center.x, center.y, center.z ); 
#endif


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
    }
    return valid_isect ; 
}




#include "csg_intersect_leaf_phicut.h"
#include "csg_intersect_leaf_thetacut.h"


/**
distance_leaf
---------------

For hints on how to implement distance functions for more primitives:

* https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
* env-;sdf-

**/

LEAF_FUNC
float distance_leaf( const float3& global_position, const CSGNode* node, const float4* plan, const qat4* itra )
{
    const unsigned typecode = node->typecode() ;  
    const unsigned gtransformIdx = node->gtransformIdx() ; 
    const bool complement = node->is_complement();

    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None

    float3 local_position  = q ? q->right_multiply(global_position,  1.f) : global_position ;  
    float distance = 0.f ;  

    switch(typecode)
    {
        case CSG_SPHERE:           distance = distance_leaf_sphere(            local_position, node->q0 )           ; break ; 
        case CSG_ZSPHERE:          distance = distance_leaf_zsphere(           local_position, node->q0, node->q1 ) ; break ; 
        case CSG_CONVEXPOLYHEDRON: distance = distance_leaf_convexpolyhedron(  local_position, node, plan )         ; break ;
        case CSG_BOX3:             distance = distance_leaf_box3(              local_position, node->q0 )           ; break ;
        case CSG_CYLINDER:         distance = distance_leaf_cylinder(          local_position, node->q0, node->q1 ) ; break ;
        case CSG_OLDCYLINDER:      distance = distance_leaf_cylinder(          local_position, node->q0, node->q1 ) ; break ;
        case CSG_PLANE:            distance = distance_leaf_plane(             local_position, node->q0 )           ; break ;
        case CSG_SLAB:             distance = distance_leaf_slab(              local_position, node->q0, node->q1 ) ; break ;
        case CSG_PHICUT:           distance = distance_leaf_phicut(            local_position, node->q0 )           ; break ;
    }

    const float sd = complement ? -distance : distance  ; 
#ifdef DEBUG
    printf("//distance_leaf typecode %d name %s complement %d sd %10.4f \n", typecode, CSG::Name(typecode), complement, sd  ); 
#endif
    return sd ; 
}


/**
intersect_leaf : must be purely single node 
----------------------------------------------

Notice that only the inverse CSG transforms are needed on the GPU as these are used to 
transform the ray_origin and ray_direction into the local origin and direction in the 
local frame of the node.   

**/

LEAF_FUNC
bool intersect_leaf( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction )
{
    const unsigned typecode = node->typecode() ;  
    const unsigned gtransformIdx = node->gtransformIdx() ; 
    const bool complement = node->is_complement();

    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None

    float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ;  
    float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;   

#ifdef DEBUG_RECORD
    printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
#endif

#ifdef DEBUG
    //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    //printf("//intersect_leaf ray_origin (%10.4f,%10.4f,%10.4f) \n",  ray_origin.x, ray_origin.y, ray_origin.z ); 
    //printf("//intersect_leaf ray_direction (%10.4f,%10.4f,%10.4f) \n",  ray_direction.x, ray_direction.y, ray_direction.z ); 
    /*
    if(q) 
    {
        printf("//intersect_leaf q.q0.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q0.f.x,q->q0.f.y,q->q0.f.z,q->q0.f.w  ); 
        printf("//intersect_leaf q.q1.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q1.f.x,q->q1.f.y,q->q1.f.z,q->q1.f.w  ); 
        printf("//intersect_leaf q.q2.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q2.f.x,q->q2.f.y,q->q2.f.z,q->q2.f.w  ); 
        printf("//intersect_leaf q.q3.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q3.f.x,q->q3.f.y,q->q3.f.z,q->q3.f.w  ); 
        printf("//intersect_leaf origin (%10.4f,%10.4f,%10.4f) \n",  origin.x, origin.y, origin.z ); 
        printf("//intersect_leaf direction (%10.4f,%10.4f,%10.4f) \n",  direction.x, direction.y, direction.z ); 
    }
    */
#endif

    bool valid_isect = false ; 
    switch(typecode)
    {
        case CSG_SPHERE:           valid_isect = intersect_leaf_sphere(           isect, node->q0,               t_min, origin, direction ) ; break ; 
        case CSG_ZSPHERE:          valid_isect = intersect_leaf_zsphere(          isect, node->q0, node->q1,     t_min, origin, direction ) ; break ; 
        case CSG_CONVEXPOLYHEDRON: valid_isect = intersect_leaf_convexpolyhedron( isect, node, plan,             t_min, origin, direction ) ; break ;
        case CSG_CONE:             valid_isect = intersect_leaf_newcone(          isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_OLDCONE:          valid_isect = intersect_leaf_oldcone(          isect, node->q0,               t_min, origin, direction ) ; break ;
        // NB: changing typecode->imp mapping is a handy way to use old imp with current geometry 
        case CSG_HYPERBOLOID:      valid_isect = intersect_leaf_hyperboloid(      isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_BOX3:             valid_isect = intersect_leaf_box3(             isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_PLANE:            valid_isect = intersect_leaf_plane(            isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_PHICUT:           valid_isect = intersect_leaf_phicut(           isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_THETACUT:         valid_isect = intersect_leaf_thetacut(         isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_SLAB:             valid_isect = intersect_leaf_slab(             isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_CYLINDER:         valid_isect = intersect_leaf_cylinder(         isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_OLDCYLINDER:      valid_isect = intersect_leaf_oldcylinder(      isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_INFCYLINDER:      valid_isect = intersect_leaf_infcylinder(      isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_DISC:             valid_isect = intersect_leaf_disc(             isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;

    }
    if(valid_isect)
    {
        if(q) q->left_multiply_inplace( isect, 0.f ) ;  
        // normals transform differently : with inverse-transform-transposed 
        // so left_multiply the normal by the inverse-transform rather than the right_multiply 
        // done above to get the inverse transformed origin and direction
        //const unsigned boundary = node->boundary();  ???

        if(complement)  // flip normal for complement 
        {
            isect.x = -isect.x ;
            isect.y = -isect.y ;
            isect.z = -isect.z ;
        }
    }   
    else
    {
         // even for miss need to signal the complement with a -0.f in isect.x
         if(complement) isect.x = -isect.x ;  
         // note that isect.y is also flipped for unbounded exit : for consumption by intersect_tree
    }


#ifdef DEBUG_RECORD
    printf("//]intersect_leaf typecode %d name %s valid_isect %d isect (%10.4f %10.4f %10.4f %10.4f)   \n", typecode, CSG::Name(typecode), valid_isect, isect.x, isect.y, isect.z, isect.w); 
#endif


    return valid_isect ; 
}

