// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>
#include "quad.h"
#include "hemi-pmt.h"

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);
// TODO: instanced analytic identity, using the above and below solid level identity buffer

rtBuffer<float4> partBuffer; 
rtBuffer<uint4>  solidBuffer; 
rtBuffer<uint4>  identityBuffer; 

// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


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
        if( P.z > q2.f.z && P.z < q3.f.z )
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
            if( P.z > q2.f.z && P.z < q3.f.z )
            { 
                if( rtPotentialIntersection( root2 ) ) 
                {
                    shading_normal = geometric_normal = -(O + root2*D)/radius; // negating as inside always(?) 
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);   // material index 0 
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
      float flip = copysignf(1.f, tmin);  // when inside: tmin < 0.f -> flip = -1.f  

      if(rtPotentialIntersection(tint))
      {
          float3 p = ray.origin + tint*ray.direction - cen_ ; 
          float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
          float pmax = fmaxf(pa);

          float3 n = make_float3(0.f);  
          if(      pmax == pa.x) n.x = copysignf( 1.f , flip*p.x ) ;              
          else if (pmax == pa.y) n.y = copysignf( 1.f , flip*p.y ) ;              
          else if (pmax == pa.z) n.z = copysignf( 1.f , flip*p.z ) ;              

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

              if(      pmax == pa.x) n.x = copysignf( 1.f , flip*p.x ) ;              
              else if (pmax == pa.y) n.y = copysignf( 1.f , flip*p.y ) ;              
              else if (pmax == pa.z) n.z = copysignf( 1.f , flip*p.z ) ;              

              shading_normal = geometric_normal = n ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
          } 
      }
  }
}






RT_PROGRAM void intersect(int primIdx)
{
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 
  const uint4& identity = identityBuffer[primIdx] ; 
  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced


  for(unsigned int p=0 ; p < numParts ; p++)
  {  
      unsigned int partIdx = solid.x + p ;  

      quad q0, q1, q2, q3 ; 

      q0.f = partBuffer[4*partIdx+0];  
      q1.f = partBuffer[4*partIdx+1];  
      q2.f = partBuffer[4*partIdx+2] ;
      q3.f = partBuffer[4*partIdx+3]; 

      switch(q2.i.w)
      {
          case 0:
                intersect_aabb(q2, q3, identity);
                break ; 
          case 1:
                intersect_zsphere<true>(q0,q1,q2,q3,identity);
                break ; 
          case 2:
                intersect_ztubs(q0,q1,q2,q3,identity);
                break ; 
          case 3:
                intersect_aabb(q2,q3,identity);
                break ; 

      }
  }

}





RT_PROGRAM void bounds (int primIdx, float result[6])
{
  // could do offline
  const uint4& solid    = solidBuffer[primIdx]; 
  unsigned int numParts = solid.y ; 

  optix::Aabb* aabb = (optix::Aabb*)result;
  *aabb = optix::Aabb();

  for(unsigned int p=0 ; p < numParts ; p++)
  { 
      unsigned int partIdx = solid.x + p ;  
      const float4& q2 = partBuffer[4*partIdx+2] ;
      const float4& q3 = partBuffer[4*partIdx+3]; 

      aabb->include( make_float3(q2), make_float3(q3) );
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

