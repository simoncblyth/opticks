// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>
#include "quad.h"
#include "hemi-pmt.h"
#include "math_constants.h"

using namespace optix;

#include "wavelength_lookup.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);
// TODO: instanced analytic identity, using the above and below solid level identity buffer

rtBuffer<float4> partBuffer; 
rtBuffer<uint4>  solidBuffer; 
rtBuffer<uint4>  identityBuffer; 
rtBuffer<float4> prismBuffer ;

// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


//#define FLT_MAX         1e30

#define DEBUG 1

/*
Ericson, Real Time Collision Detection p196-198



   Ray

         L(t) = A + t(B - A)   


      A ----------------------------------B



   Cylinder

            d = Q - P               axis
            
            v = X - P               surface vec

                  v.d
            w =  ----- d            component of v along axis
                  d.d 

          r*r = (v - w).(v - w)     surface locus      


                                            B
                                           /
                    +--------Q--------+   /
                    |        |        |  /
                    |        |        | /
                    |        |        |/ 
                    |        |        * 
                    |        |       /| 
                    |        |      / |      
                    |        |     /  |
                    |     d  |    /   |         
                    |        |   /    |    
                    |        |  /     |       
                    |        | /      |
                    |        |/       | 
                    |        /.  .  . | .  .  .  .  .  . 
                    |       /|        |                
                    |      / |        |                .
                    |     /  |        | 
                    |    /   |        |                .
                    |   /    |        | 
                    |  /     |        |                .
                    | /      |        | 
                    |/       |        |                .
                    I        |        | 
                   /|        |        |                .
                  / |        |        | 
                 /  |        |        |               n.d
                /   +--------P--------+
          n    /            .|   r                     .
              /           .  
             /          .    |                         .
            /         .     
           /        .        |                         .
          /       . 
         /      .  m         |   m.d                   .
        /     .             
       /    .                |                         . 
      /   .  
     /  .                    |                         .
    / .    
   A .  .   .  .   .   .   . | .  .  .  .  .  .   .    .    



    
     Normal at intersection point I is component of I-P 
     with the axial component subtracted

        (I-P) - (I-P).(Q-P)


       v = L(t) - P 

         = (A - P) +  t(B - A)


      v  =   m + t n

      m  = A - P           ray origin in cylinder frame

      m.d                  axial coordinate of ray origin 

      n  = B - A           ray direction



                  v.d
            w =  ----- d            component of v along axis
                  d.d 


                  m.d + t n.d
            w =  -------------- d      
                      d.d 

          r*r = (v - w).(v - w)   

          r*r  = v.v + w.w - 2 v.w  


          v.v = ( m + t n ).(m + t n)

              = m.m + 2t m.n + t*t n.n 



   Intersection with P endcap plane 

       (X - P).d = 0 

       ( A + t (B - A) - P).d = 0 

       (  m + t n ).d = 0        =>   t = - m.d / n.d          

                when axial n in d direction          

                                     t  = - m.n / n.n    
      radial requirement 

         (m + t n).(m + t n) < rr 

         mm - rr + 2t m.n + t*t nn < 0 


   Intersection with Q endcap plane 

       (X - Q).d = 0      Q = d + P  

       (A + t (B - A) - Q).d = 0 
 
       ( A - P + t (B - A) - d ).d = 0 

       (  m + t n - d ).d = 0      =>    t = ( d.d - m.d ) / n.d

                when axial n in d direction          


      radial requirement 

         (m + t n - d).(m + t n - d) < rr 

         mm + tt nn + dd  





*/

enum
{
    ENDCAP_P = 0x1 <<  0,    
    ENDCAP_Q = 0x1 <<  1
};    
 


static __device__
void intersect_ztubs(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity )
{
/* 
Position shift below is to match between different cylinder Z origin conventions

* Ericson calc implemented below has cylinder origin at endcap P  
* detdesc/G4 Tubs has cylinder origin in the center 

*/
    float sizeZ = q1.f.x ; 
    float z0 = q0.f.z - sizeZ/2.f ;     
    float3 position = make_float3( q0.f.x, q0.f.y, z0 );  // 0,0,-169.
    float clipped_sizeZ = q3.f.z - q2.f.z ;  
 
    float radius = q0.f.w ;
    int flags = q1.i.w ; 

    bool PCAP = flags & ENDCAP_P ; 
    bool QCAP = flags & ENDCAP_Q ;

    //rtPrintf("intersect_ztubs position %10.4f %10.4f %10.4f \n", position.x, position.y, position.z );
    //rtPrintf("intersect_ztubs flags %d PCAP %d QCAP %d \n", flags, PCAP, QCAP);
 
    float3 m = ray.origin - position ;
    float3 n = ray.direction ; 
    float3 d = make_float3(0.f, 0.f, clipped_sizeZ ); 

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
        if(c > 0.f) return ;    // ray starts and ends outside cylinder
        if(md < 0.f && PCAP)    // ray origin on P side
        {
            float t = -mn/nn ;  // P endcap 
            if( rtPotentialIntersection(t) )
            {
                shading_normal = geometric_normal = -dnorm  ;  
                instanceIdentity = identity ; 
#ifdef DEBUG
                instanceIdentity.y = HP_PAXI_O ;
#endif

                rtReportIntersection(0);
            }
        } 
        else if(md > dd && QCAP) // ray origin on Q side 
        {
            float t = (nd - mn)/nn ;  // Q endcap
            if( rtPotentialIntersection(t) )
            {
                shading_normal = geometric_normal = dnorm ; 
                instanceIdentity = identity ; 
#ifdef DEBUG
                instanceIdentity.y = HP_QAXI_O ;
#endif
                rtReportIntersection(0);
            }
        }
        else    // md 0:dd, ray origin inside 
        {
            if( nd > 0.f && PCAP) // ray along +d 
            {
                float t = -mn/nn ;    // P endcap from inside
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = dnorm  ;  
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_PAXI_I ;
#endif
                    rtReportIntersection(0);
                }
            } 
            else if(QCAP)  // ray along -d
            {
                float t = (nd - mn)/nn ;  // Q endcap from inside
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = -dnorm ; 
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_QAXI_I ;
#endif
                    rtReportIntersection(0);
                }
            }
        }
        return ;   // hmm 
    }

    if(disc > 0.0f)  // intersection with the infinite cylinder
    {
        float sdisc = sqrtf(disc);

        float root1 = (-b - sdisc)/a;     
        float ad1 = md + root1*nd ;        // axial coord of intersection point 
        float3 P1 = ray.origin + root1*ray.direction ;  

        if( ad1 > 0.f && ad1 < dd )  // intersection inside cylinder range
        {
            if( rtPotentialIntersection(root1) ) 
            {
                float3 N  = (P1 - position)/radius  ;  
                N.z = 0.f ; 

                //rtPrintf("intersect_ztubs r %10.4f disc %10.4f sdisc %10.4f root1 %10.4f P %10.4f %10.4f %10.4f N %10.4f %10.4f \n", 
                //    radius, disc, sdisc, root1, P1.x, P1.y, P1.z, N.x, N.y );

                shading_normal = geometric_normal = normalize(N) ;
                instanceIdentity = identity ; 
#ifdef DEBUG
                instanceIdentity.y = HP_WALL_O ;
#endif
                rtReportIntersection(0);
            } 
        } 
        else if( ad1 < 0.f && PCAP ) //  intersection outside cylinder on P side
        {
            if( nd <= 0.f ) return ; // ray direction away from endcap
            float t = -md/nd ;   // P endcap 
            float checkr = k + t*(2.f*mn + t*nn) ; // bracket typo in book 2*t*t makes no sense   
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = -dnorm  ;  
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_PCAP_O ;
#endif
                    rtReportIntersection(0);
                }
            } 
        } 
        else if( ad1 > dd && QCAP  ) //  intersection outside cylinder on Q side
        {
            if( nd >= 0.f ) return ; // ray direction away from endcap
            float t = (dd-md)/nd ;   // Q endcap 
            float checkr = k + dd - 2.0f*md + t*(2.f*(mn-nd)+t*nn) ;             
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = dnorm  ;  
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_QCAP_O ;
#endif

                    rtReportIntersection(0);
                }
            } 
        }


        float root2 = (-b + sdisc)/a;     // far root : means are inside (always?)
        float ad2 = md + root2*nd ;        // axial coord of far intersection point 
        float3 P2 = ray.origin + root2*ray.direction ;  


        if( ad2 > 0.f && ad2 < dd )  // intersection from inside against wall 
        {
            if( rtPotentialIntersection(root2) ) 
            {
                float3 N  = (P2 - position)/radius  ;  
                N.z = 0.f ; 

                shading_normal = geometric_normal = -normalize(N) ;
                instanceIdentity = identity ; 
#ifdef DEBUG
                instanceIdentity.y = HP_WALL_I ;
#endif
                rtReportIntersection(0);
            } 
        } 
        else if( ad2 < 0.f && PCAP ) //  intersection from inside to P endcap
        {
            float t = -md/nd ;   // P endcap 
            float checkr = k + t*(2.f*mn + t*nn) ; // bracket typo in book 2*t*t makes no sense   
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = dnorm  ;  
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_PCAP_I ;
#endif
                    rtReportIntersection(0);
                }
            } 
        } 
        else if( ad2 > dd  && QCAP ) //  intersection from inside to Q endcap
        {
            float t = (dd-md)/nd ;   // Q endcap 
            float checkr = k + dd - 2.0f*md + t*(2.f*(mn-nd)+t*nn) ;             
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = -dnorm  ;  
                    instanceIdentity = identity ; 
#ifdef DEBUG
                    instanceIdentity.y = HP_QCAP_I ;
#endif
                    rtReportIntersection(0);
                }
            } 
        }



    }
}

/*

    Ray-Sphere
    ~~~~~~~~~~~~~

    Ray(xyz) = ori + t*dir     dir.dir = 1

    (t*dir + ori-cen).(t*dir + ori-cen) = rad^2

     t^2 dir.dir + 2t(ori-cen).dir + (ori-cen).(ori-cen) - rad^2 = 0  

     t^2 + 2t O.D + O.O - radius = 0 

     a t^2 + b t + c = 0  =>   t = ( -b +- sqrt(b^2 - 4ac) )/2a 


        t = -2 O.D +-  sqrt(4* [(b/2*b/2) - (O.O - rad*rad)])
            ----------------------------------------- 
                            2

          =   - O.D +- sqrt(  O.D*O.D - (O.O - rad*rad) ) 


      normal to sphere at intersection point  (O + t D)/radius

            (ori + t D) - cen
            ------------------
                  radius

 


*/




/*

*intersect_zsphere*
     shooting millions of photons at the +Z pole (Pyrex sphere in vacuum) 
     from a hemi-spherical torch source leads to ~5% level thinking already in
     pyrex when actually in vacuum (ie they failed to intersect at the targetted +Z pole)
     Visible as a red Pyrex leak within the white Vacuum outside the sphere.

     Problem confirmed to be due to bbox effectively clipping the sphere pole
     (and presumably all 6 points where bbox touches the sphere are clipped) 

     Avoid issue by slightly increasing bbox size by factor ~ 1+1e-6


Alternate sphere intersection using geometrical 
rather than algebraic approach, starting from 
t(closest approach) to sphere center

* http://www.vis.uky.edu/~ryang/teaching/cs535-2012spr/Lectures/13-RayTracing-II.pdf



RESOLVED : tangential incidence artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PILOT ERROR: the sphere was offset at (-1,1,0) when should have been at origin, hangover from leak avoidance?** 

Intersecting a disc of parallel rays of the same radius as a sphere
causes artifacts at tangential incidence (irregular bunched heart shape of reflected photons), 
changing radius of disc to avoid tangentials (eg radius of sphere 100, disc 95) avoids the 
issue.


*/

template<bool use_robust_method>
static __device__
void intersect_zsphere(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
{

  float3 center = make_float3(q0.f);
  float radius = q0.f.w;

  float3 O = ray.origin - center;
  float3 D = ray.direction;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;

 /*
  rtPrintf("intersect_sphere %10.4f %10.4f %10.4f : %10.4f disc %10.4f \n", 
       center.x,  
       center.y,  
       center.z,  
       radius,
       disc);  
  */

    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool do_refine = use_robust_method && fabsf(root1) > 10.f * radius ;  // long ray 

        if(do_refine) // refine root1
        {
            float3 O1 = O + root1 * ray.direction;  //  move origin along to 1st intersection point
            b = dot(O1, D);
            c = dot(O1, O1) - radius*radius;
            disc = b*b - c;
            if(disc > 0.0f) 
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }
        float3 P = ray.origin + (root1 + root11)*ray.direction ;  
        bool check_second = true;
        if( P.z >= q2.f.z && P.z <= q3.f.z )
        {
            if( rtPotentialIntersection( root1 + root11 ) ) 
            {
                shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
                instanceIdentity = identity ; 
                if(rtReportIntersection(0)) check_second = false;
            } 
        }

        if(check_second) 
        {
            float root2 = (-b + sdisc) + (do_refine ? root11 : 0.f);   // unconfirmed change root1 -> root11
            P = ray.origin + root2*ray.direction ;  
            if( P.z >= q2.f.z && P.z <= q3.f.z )
            { 
                if( rtPotentialIntersection( root2 ) ) 
                {
                    shading_normal = geometric_normal = (O + root2*D)/radius; 
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);   // material index 0 

                    // NB: **NOT** negating normal when inside as that 
                    //     would break rules regards "solidity" of geometric normals
                    //     normal must depend on geometry at intersection point **only**, 
                    //     with no dependence on ray direction
                }
            }
        }
    }
}


/*

Ray Box Intersection Slab Method
==================================

* http://tavianator.com/fast-branchless-raybounding-box-intersections/


                   |            |  / 
                   |            | /
                   |            |/ 
                   |            $ 
                   |           /|
                   |          / |
                   |         /  |
                   |        /   |
                   |       /    |
                   |      /     |
          .        |     /      |
          .        |    /       |
          .        |   /        | 
          .        |  /         |
   +      +--------+-$:B--------X----------- bbmax y plane
   |      .        |/           |
   |      .      A:@            |
   |      .       /|     C      |
   |      .      / |            |
 t1.y     .     /  |            |
   |  +   +----@---N------------+------------ bbmin y plane
   |  |   .   /    |            |
   | t0.y .  /     |            |
   |  |   . /      |            |
   |  |   ./       |            |
...+..+.+.O........+............+.........
         /.                      
        / .                      
       /  +--t0.x--+             
          .                      
          +---------t1.x--------+
                                 
          +-near.x-+

          +---------far.x-------+



         O : ray.origin    (t = 0)

         X : bb.max

         N : bb.min

         C : bb.center (min+max)/2

         @ : intersections with the bbmin planes

         $ : intersections with the bbmax planes



      near : min(t0, t1)   [ min(t0.x, t1.x), min(t0.y,t1.y), min(t0.z,t1.z) ]
      far  : max(t0, t1)   [ max(t0.x, t1.x), max(t0.y,t1.y), max(t0.z,t1.z) ] 

             ordering of slab intersections for each axis, into nearest and furthest 

             ie: pick either bbmin or bbmax planes for each axis 
             depending on the direction the ray is coming from

             think about looking at a box from multiple viewpoints 
             

      tmin : max(near)

             pick near slab intersection with the largest t value, 
             ie furthest along the ray, so this picks "A:@" above
             rather than the "@" slab(not-box) intersection

      tmax : min(far)

             pick far slab intersection with the smallest t value,
             ie least far along the ray, so this picks "$:B" above
             rather than the "$" slab(not-box) intersection 

      tmin <= tmax 

             means the ray has a segment within the box, ie intersects with it, 
             but need to consider different cases:


      tmin <= tmax && tmin > 0 

                       |        |
              ---O->---+--------+----- 
                       |        | 
                     tmin     tmax 

             ray.origin outside the box
             
             intersection at t = tmin


      tmin <= tmax && tmin < 0 

                       |        |
                 ------+---O->--+----- 
                       |        | 
                     tmin     tmax 

             ray.origin inside the box, so there is intersection in direction
             opposite to ray.direction (behind the viewpoint) 

             intersection at t = tmax


      tmin <= tmax && tmax < 0    (implies tmin < 0 too)

                       |        |
                 ------+--------+---O->-- 
                       |        | 
                     tmin     tmax 


             ray.origin outside the box, with intersection "behind" the ray 
             so must disqualify the intersection

      
       tmin <= tmax && tmax > 0 
 
             qualifying intersection condition, with intersection at 

                    tmin > 0 ? tmin : tmax

             where tmin < 0 means are inside



       is check_second needed ?

             YES for rendering (not for propagation, but does no harm?) 
             this handles OptiX epsilon near clipping,
             a way to clip ray trace rendered geometry so can look inside
             in this situation valid front face box intersections 
             will be disqualifies so need to fall back to the other intersection


       Normals at intersections will be in one of six directions: +x -x +y -y +z -z 

             http://graphics.ucsd.edu/courses/cse168_s06/ucsd/CSE168_raytrace.pdf

       Consider vector from box center to intersection point 
       ie intersect coordinates in local box frame
       Think of viewing unit cube at origin from different
       directions (eg face on down the axes).

       Determine which face from the largest absolute 
       value of (x,y,z) of local box frame intersection point. 

       Normal is defined to be in opposite direction
       to the impinging ray. 

       * **WRONG : NOT AT INTERSECT LEVEL
       * INTERSECT LEVEL IS FOR DEFINING GEOMETRY, NOT THE RELATIONSHIP 
         OF VIEWPOINT AND GEOEMTRY : THAT COMES LATER
          
 
                  +---@4------+
                  |   /\      |
                  |   +       |
             +->  @1     +->  @2  
                  |   +       |
                  |   \/      @3   <-+
                  +---@5--@6--+
                          /\
                          + 

                               normal

            @1   [-1,0,0]    =>  [-1,0,0]   ( -x from outside, no-flip )     

            @2   [ 1,0,0]        [-1,0,0]   ( +x from inside, flipped ) 

            @3   [ 1,-0.7,0] =>  [ 1,0,0]   ( +x from outside, no-flip ) 

            @4   [-0.5,1,0]  =>  [ 0,-1,0]   ( -y from inside, flipped )

            @5   [-0.5,-1,0] =>  [ 0, 1,0]   ( +y from inside, flipped )

            @6   [ 0.5,-1,0] =>  [ 0,-1,0]   ( -y from outside, no-flip)



      RULES FOR GEOMETRIC NORMALS  
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      * MUST BE RIGIDLY AND CONSISTENTLY "ATTACHED" TO GEOMETRY DEPENDING ONLY ON
        THE GEOMETRY AT THE INTERSECTION POINT AND SOME FIXED CONVENTION 
        (CONSIDER TRIANGLE INTERSECTION EXAMPLE)

      * I ADOPT STANDARD CONVENTION OF OUTWARDS POINTED NORMALS : 
        SO IT SHOULD BE DARK INSIDE BOXES

      THIS MEANS:

      **DO NOT FLIP BASED ON WHERE THE RAYS ARE COMING FROM OR BEING INSIDE BOX**   


      Edge Cases
      ~~~~~~~~~~~~

      http://tavianator.com/fast-branchless-raybounding-box-intersections/
      http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/ 
 
      Floating-point infinity handling should correctly deal 
      with axis aligned ray directions but there is literally an edge case 
      when the ray starts exactly on the edge the box 
    
          0 * inf = nan which occurs when the ray starts 
  




*/

static __device__
void intersect_aabb(quad& q2, quad& q3, const uint4& identity)
{
  const float3 min_ = make_float3(q2.f.x, q2.f.y, q2.f.z); 
  const float3 max_ = make_float3(q3.f.x, q3.f.y, q3.f.z); 
  const float3 cen_ = 0.5f*(min_ + max_) ;    

  float3 t0 = (min_ - ray.origin)/ray.direction;
  float3 t1 = (max_ - ray.origin)/ray.direction;

  // slab method 
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );


  if(tmin <= tmax && tmax > 0.f) 
  {
      bool check_second = true;
      float tint = tmin > 0.f ? tmin : tmax ; 

      if(rtPotentialIntersection(tint))
      {
          float3 p = ray.origin + tint*ray.direction - cen_ ; 
          float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
          float pmax = fmaxf(pa);

          float3 n = make_float3(0.f) ;
          if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
          else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
          else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              


          shading_normal = geometric_normal = n ;
          instanceIdentity = identity ;
          if(rtReportIntersection(0)) check_second = false ;   // material index 0 
      } 

      // handle when inside box, or are epsilon near clipped 
      if(check_second)
      {
          if(rtPotentialIntersection(tmax))
          {
              float3 p = ray.origin + tmax*ray.direction - cen_ ; 
              float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
              float pmax = fmaxf(pa);


              /*
              float3 n = make_float3(0.f);  

              if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
              else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
              else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              
              */

              float3 n = make_float3(1.f,0.f,0.f);  


              shading_normal = geometric_normal = n ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
          } 
      }
  }
}








static __device__
void intersect_box(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
{

  const float3 min_ = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
  const float3 max_ = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
  const float3 cen_ = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

  float3 t0 = (min_ - ray.origin)/ray.direction;
  float3 t1 = (max_ - ray.origin)/ray.direction;

  // slab method 
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax && tmax > 0.f) 
  {
      bool check_second = true;
      float tint = tmin > 0.f ? tmin : tmax ; 

      if(rtPotentialIntersection(tint))
      {
          float3 p = ray.origin + tint*ray.direction - cen_ ; 
          float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
          float pmax = fmaxf(pa);

          float3 n = make_float3(0.f) ;
          if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
          else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
          else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              


          shading_normal = geometric_normal = n ;
          instanceIdentity = identity ;
          if(rtReportIntersection(0)) check_second = false ;   // material index 0 
      } 

      // handle when inside box, or are epsilon near clipped 
      if(check_second)
      {
          if(rtPotentialIntersection(tmax))
          {
              float3 p = ray.origin + tmax*ray.direction - cen_ ; 
              float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
              float pmax = fmaxf(pa);

              float3 n = make_float3(0.f);  

              if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
              else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
              else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

              shading_normal = geometric_normal = n ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
          } 
      }
  }
}












// from tutorial9 intersect_chull
static __device__
void intersect_prism(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
{
  int nplane = 5 ;

  float t0 = -CUDART_INF_F ; 
  float t1 =  CUDART_INF_F ; 

  float3 t0_normal = make_float3(0.f);
  float3 t1_normal = make_float3(0.f);

  for(int i = 0; i < nplane && t0 < t1 ; ++i ) 
  {
    float4 plane = prismBuffer[i];
    float3 n = make_float3(plane);
    float  d = plane.w;

    float denom = dot(n, ray.direction);
    if(denom == 0.f) continue ;   
    float t = -(d + dot(n, ray.origin))/denom;
    
    // Avoiding infinities.
    // Somehow infinities arising from perpendicular other planes
    // prevent normal incidence plane intersection.
    // This caused a black hairline crack around the prism middle. 
    //
    // In aabb slab method infinities were well behaved and
    // did not change the result, but not here.
    //
    // BUT: still getting extended edge artifact when view from precisely +X+Y
    // http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
    //
    
    if( denom < 0.f){  // ray opposite to normal, ie ray from outside entering
      if(t > t0){
        t0 = t;
        t0_normal = n;
      }
    } else {          // ray same hemi as normal, ie ray from inside exiting 
      if(t < t1){
        t1 = t;
        t1_normal = n;
      }
    }

  }

  if(t0 > t1)
    return;

  if(rtPotentialIntersection( t0 )){
    shading_normal = geometric_normal = t0_normal;
    instanceIdentity = identity ;
    rtReportIntersection(0);
  } else if(rtPotentialIntersection( t1 )){
    shading_normal = geometric_normal = t1_normal;
    instanceIdentity = identity ;
    rtReportIntersection(0);
  }
}


/*
Ray-plane intersection 

     n.(O + t D - p) = 0 

     n.O + t n.D - n.p = 0

           t = (n.p - n.O)/n.D  
           t = ( -d - n.O)/n.D
             = -(d + n.O)/n.D

   n.D = 0   when ray direction in plane of face

  why is +Z problematic  n = (0,0,1)

  axial direction rays from +X, +Y, +Z and +X+Y are 
  failing to intersect the prism.  

  Breaking axial with a small delta 0.0001f 
  avoids the issue. 


  See a continuation of edge artifact when viewing from precisely +X+Y


*/






static __device__
float4 make_plane( float3 n, float3 p ) 
{
  n = normalize(n);
  float d = -dot(n, p); 
  return make_float4( n, d );
}

/*

http://mathworld.wolfram.com/Plane.html

    n.(x - p) = 0    normal n = (a,b,c), point in plane p

    n.x - n.p = 0

    ax + by + cz + d = 0        d = -n.p

+Z face of unit cube

    n = (0,0,1)
    p = (0,0,1)
    d = -n.p = -1   ==> z + (-1)  = 0,     z = 1     

-Z face of unit cube

    n = (0,0,-1)
    p = (0,0,-1)
    d = -n.p = -1   ==>  (-z) + (-1) = 0,    z = -1   

*/


static __device__
void make_prism( const float4& param, optix::Aabb* aabb ) 
{
/*
 Mid line of the symmetric prism spanning along z from -depth/2 to depth/2

                                                 
                            A  (0,height,0)     Y
                           /|\                  |
                          / | \                 |
                         /  |  \                +---- X
                        /   h   \              Z  
                       /    |    \ (x,y)   
                      M     |     N   
                     /      |      \
                    L-------O-------R   
         (-hwidth,0, 0)           (hwidth, 0, 0)


    For apex angle 90 degrees, hwidth = height 

*/


    float angle  = param.x > 0.f ? param.x : 90.f ; 
    float height = param.y > 0.f ? param.y : param.w  ;
    float depth  = param.z > 0.f ? param.z : param.w  ;

    float hwidth = height*tan((M_PIf/180.f)*angle/2.0f) ;   

    rtPrintf("make_prism angle %10.4f height %10.4f depth %10.4f hwidth %10.4f \n", angle, height, depth, hwidth);

    float ymax =  height/2.0f ;   
    float ymin = -height/2.0f ;   

    float3 apex = make_float3( 0.f, ymax,  0.f );
    float3 base = make_float3( 0.f, ymin,  0.f) ; 
    float3 front = make_float3(0.f, ymin,  depth/2.f) ; 
    float3 back  = make_float3(0.f, ymin, -depth/2.f) ; 

    prismBuffer[0] = make_plane( make_float3(  height, hwidth,  0.f), apex  ) ;  // +X+Y 
    prismBuffer[1] = make_plane( make_float3( -height, hwidth,  0.f), apex  ) ;  // -X+Y 
    prismBuffer[2] = make_plane( make_float3(     0.f,  -1.0f,  0.f), base  ) ;  //   -Y 
    prismBuffer[3] = make_plane( make_float3(     0.f,    0.f,  1.f), front ) ;  //   +Z
    prismBuffer[4] = make_plane( make_float3(     0.f,    0.f, -1.f), back  ) ;  //   -Z

    rtPrintf("make_prism plane[0] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[0].x, prismBuffer[0].y, prismBuffer[0].z, prismBuffer[0].w );
    rtPrintf("make_prism plane[1] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[1].x, prismBuffer[1].y, prismBuffer[1].z, prismBuffer[1].w );
    rtPrintf("make_prism plane[2] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[2].x, prismBuffer[2].y, prismBuffer[2].z, prismBuffer[2].w );
    rtPrintf("make_prism plane[3] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[3].x, prismBuffer[3].y, prismBuffer[3].z, prismBuffer[3].w );
    rtPrintf("make_prism plane[4] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[4].x, prismBuffer[4].y, prismBuffer[4].z, prismBuffer[4].w );

    float3 max = make_float3( hwidth, ymax,  depth/2.f);
    float3 min = make_float3(-hwidth, ymin, -depth/2.f);
    float3 eps = make_float3( 0.001f );

    aabb->include( min - eps, max + eps );

/*
make_prism angle    90.0000 height   200.0000 depth   200.0000 hwidth   200.0000 
make_prism plane[0]     0.7071     0.7071     0.0000  -141.4214 
make_prism plane[1]    -0.7071     0.7071     0.0000  -141.4214 
make_prism plane[2]     0.0000    -1.0000     0.0000    -0.0000 
make_prism plane[3]     0.0000     0.0000     1.0000  -100.0000 
make_prism plane[4]     0.0000     0.0000    -1.0000  -100.0000 
*/

}




RT_PROGRAM void bounds (int primIdx, float result[6])
{
  // could do offline
  // but this is great place to dump things checking GPU side state
  // as only run once


  source_check(); 

  const uint4& solid    = solidBuffer[primIdx]; 
  uint4 identity = identityBuffer[instance_index] ; 
  unsigned int numParts = solid.y ; 

  optix::Aabb* aabb = (optix::Aabb*)result;
  *aabb = optix::Aabb();
  // expand aabb to include all the bbox of the parts 

  for(unsigned int p=0 ; p < numParts ; p++)
  { 
      unsigned int partIdx = solid.x + p ;  

      quad q0, q1, q2, q3 ; 

      q0.f = partBuffer[4*partIdx+0];  
      q1.f = partBuffer[4*partIdx+1];  
      q2.f = partBuffer[4*partIdx+2] ;
      q3.f = partBuffer[4*partIdx+3]; 
      
      int partType = q2.i.w ; 

      identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
      unsigned int boundary = q1.u.z ; 
      rtPrintf("bounds primIdx %u p %u partIdx %u boundary %u identity (%u,%u,%u,%u) partType %d \n", primIdx, p, partIdx, boundary,  
                  identity.x, 
                  identity.y, 
                  identity.z, 
                  identity.w,
                  partType 
              );  


      rtPrintf("q0 %10.4f %10.4f %10.4f %10.4f q1 %10.4f %10.4f %10.4f %10.4f \n",
                  q0.f.x, 
                  q0.f.y, 
                  q0.f.z, 
                  q0.f.w,
                  q1.f.x, 
                  q1.f.y, 
                  q1.f.z, 
                  q1.f.w);

      rtPrintf("q2 %10.4f %10.4f %10.4f %10.4f q3 %10.4f %10.4f %10.4f %10.4f \n",
                  q2.f.x, 
                  q2.f.y, 
                  q2.f.z, 
                  q2.f.w,
                  q3.f.x, 
                  q3.f.y, 
                  q3.f.z, 
                  q3.f.w);


      if(partType == 4) 
      {
          make_prism(q0.f, aabb) ;
      }
      else
      {
          aabb->include( make_float3(q2.f), make_float3(q3.f) );
      }
  } 

  rtPrintf("bounds primIdx %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, 
       result[0],
       result[1],
       result[2],
       result[3],
       result[4],
       result[5]
     );

}



RT_PROGRAM void intersect(int primIdx)
{
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 

  //const uint4& identity = identityBuffer[primIdx] ; 
  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced

  // try with just one identity per-instance 
  uint4 identity = identityBuffer[instance_index] ; 


  for(unsigned int p=0 ; p < numParts ; p++)
  {  
      unsigned int partIdx = solid.x + p ;  

      quad q0, q1, q2, q3 ; 

      q0.f = partBuffer[4*partIdx+0];  
      q1.f = partBuffer[4*partIdx+1];  
      q2.f = partBuffer[4*partIdx+2] ;
      q3.f = partBuffer[4*partIdx+3]; 

      identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)

      int partType = q2.i.w ; 

      // TODO: use enum
      switch(partType)
      {
          case 0:
                intersect_aabb(q2, q3, identity);
                break ; 
          case 1:
                intersect_zsphere<false>(q0,q1,q2,q3,identity);
                break ; 
          case 2:
                intersect_ztubs(q0,q1,q2,q3,identity);
                break ; 
          case 3:
                intersect_box(q0,q1,q2,q3,identity);
                break ; 
          case 4:
                intersect_prism(q0,q1,q2,q3,identity);
                break ; 

      }
  }

}


