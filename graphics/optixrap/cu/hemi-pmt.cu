// based on /usr/local/env/cuda/OptiX_380_sdk/julia/sphere.cu

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);

rtBuffer<float4> partBuffer; 
rtBuffer<uint4>  solidBuffer; 
rtBuffer<uint4>  identityBuffer; 


// attributes communicate to closest hit program,
// they must be set inbetween rtPotentialIntersection and rtReportIntersection

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



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
                    *        |        | 
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

static __device__
void intersect_ztubs(const float4& zrg, const float4& q0, const float4& q1, const uint4& identity )
{
    float3 position = make_float3( q0 ); 
    float radius = q0.w;
    float sizeZ = q1.x ; 

    float3 m = ray.origin - position ;
    float3 n = ray.direction ; 
    float3 d = make_float3(0.f, 0.f, sizeZ ); 

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

    // axial ray 
    if(fabs(a) < 1e-6f)     
    {
        if(c > 0.f) return ; // ray starts and ends outside cylinder
        if(md < 0.f)         // ray origin on P side
        {
            float t = -mn/nn ;    // P endcap 
            if( rtPotentialIntersection(t) )
            {
                shading_normal = geometric_normal = -dnorm  ;  
                instanceIdentity = identity ; 
                rtReportIntersection(0);
            }
        } 
        else if(md > dd) // ray origin on Q side 
        {
            float t = (nd - mn)/nn ;  // Q endcap
            if( rtPotentialIntersection(t) )
            {
                shading_normal = geometric_normal = dnorm ; 
                instanceIdentity = identity ; 
                rtReportIntersection(0);
            }
        }
        else    // md 0:dd, ray origin inside 
        {
            if( nd > 0.f ) // ray along +d 
            {
                float t = -mn/nn ;    // P endcap from inside
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = dnorm  ;  
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);
                }
            } 
            else   // ray along -d
            {
                float t = (nd - mn)/nn ;  // Q endcap from inside
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = -dnorm ; 
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);
                }
            }
        }
        return ; 
    }

    if(disc > 0.0f)  // intersection with the infinite cylinder
    {
        float sdisc = sqrtf(disc);

        float root1 = (-b - sdisc)/a;     // what about other root ? 
        float ad1 = md + root1*nd ;        // axial coord of intersection point 
        float3 P1 = ray.origin + root1*ray.direction ;  

        if( ad1 > 0.f && ad1 < dd )  // intersection inside cylinder range
        {
            if( rtPotentialIntersection(root1) ) 
            {
                float3 N  = (P1 - position)/radius  ;  
                N.z = 0.f ; 

                rtPrintf("intersect_ztubs r %10.4f disc %10.4f sdisc %10.4f root1 %10.4f P %10.4f %10.4f %10.4f N %10.4f %10.4f \n", 
                    radius, disc, sdisc, root1, P1.x, P1.y, P1.z, N.x, N.y );

                shading_normal = geometric_normal = normalize(N) ;
                instanceIdentity = identity ; 
                rtReportIntersection(0);
            } 
        } 
        else if( ad1 < 0.f ) //  intersection outside cylinder on P side
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
                    rtReportIntersection(0);
                }
            } 
        } 
        else if( ad1 > dd  ) //  intersection outside cylinder on Q side
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
                rtReportIntersection(0);
            } 
        } 
        else if( ad2 < 0.f ) //  intersection from inside to P endcap
        {
            float t = -md/nd ;   // P endcap 
            float checkr = k + t*(2.f*mn + t*nn) ; // bracket typo in book 2*t*t makes no sense   
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = dnorm  ;  
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);
                }
            } 
        } 
        else if( ad2 > dd  ) //  intersection from inside to Q endcap
        {
            float t = (dd-md)/nd ;   // Q endcap 
            float checkr = k + dd - 2.0f*md + t*(2.f*(mn-nd)+t*nn) ;             
            if ( checkr < 0.f )
            {
                if( rtPotentialIntersection(t) )
                {
                    shading_normal = geometric_normal = -dnorm  ;  
                    instanceIdentity = identity ; 
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


     Ray-Cylinder
     ~~~~~~~~~~~~~

         
          \
           \
     +----+-\--+
     |    |  \ |
     |    |   \|
     +----A----P--> normal   
     |    |   /|\
     |    |  / | \
     |    | /  |  \  ray        
     |    |/   |   \
     +----C----+    \
             r 
         
     Cylinder position C, axis C-A, intersection with ray at P

    
     Normal at intersection point is component of P-C 
     with the axial component subtracted

        (P-C) - (P-C).(A-C)


     float3 I = (P - position)/radius ; // cylinder intersection point in cylinder frame and radius units   



*/

template<bool use_robust_method>
static __device__
void intersect_zsphere(const float4& zrg, const float4& q0, const uint4& identity)
{
  float3 center = make_float3(q0);
  float radius = q0.w;

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
        if( P.z > zrg.x && P.z < zrg.y )
        {
            bool check_second = true;
            if( rtPotentialIntersection( root1 + root11 ) ) 
            {
                shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
                instanceIdentity = identity ; 
                if(rtReportIntersection(0)) check_second = false;
            } 
            if(check_second) 
            {
                float root2 = (-b + sdisc) + (do_refine ? root11 : 0.f);   // unconfirmed change root1 -> root11
                P = ray.origin + root2*ray.direction ;  
                if( P.z > zrg.x && P.z < zrg.y )
                { 
                    if( rtPotentialIntersection( root2 ) ) 
                    {
                        shading_normal = geometric_normal = (O + root2*D)/radius;
                        instanceIdentity = identity ; 
                        rtReportIntersection(0);   // material index 0 
                    }
                }
            }
        }
    }
}

static __device__
void intersect_aabb(const float4& q2, const float4& q3, const uint4& identity)
{
  const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
  const float3 max_ = make_float3(q3.x, q3.y, q3.z); 

  float3 t0 = (min_ - ray.origin)/ray.direction;
  float3 t1 = (max_ - ray.origin)/ray.direction;

  // slab method 
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  float3 n = make_float3(0.f);  

  if(tmin <= tmax) 
  {
      if(rtPotentialIntersection(tmin))
      {
          // hmm what about inside box ?
          if(     tmin == near.x) n.x = 1. ;
          else if(tmin == near.y) n.y = 1. ;
          else if(tmin == near.z) n.z = 1. ;

          shading_normal = geometric_normal = n ;
          instanceIdentity = identity ;
          rtReportIntersection(0);   // material index 0 
      } 
  }
}

static __device__
bool intersect_aabb(const float4& q2, const float4& q3)
{
    const float3 min_ = make_float3(q2.x, q2.y, q2.z); 
    const float3 max_ = make_float3(q3.x, q3.y, q3.z); 
    float3 t0 = (min_ - ray.origin)/ray.direction;
    float3 t1 = (max_ - ray.origin)/ray.direction;
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf( near );
    float tmax = fminf( far );
    return tmin <= tmax ;
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

      const float4& q0 = partBuffer[4*partIdx+0];  
      const float4& q1 = partBuffer[4*partIdx+1];  
      const float4& q2 = partBuffer[4*partIdx+2] ;
      const float4& q3 = partBuffer[4*partIdx+3]; 

      float4 zrange = make_float4( q2.z , q3.z, 0.f, 0.f ) ;
      int typecode = __float_as_int(q2.w); 

      switch(typecode)
      {
          case 0:
                intersect_aabb(q2, q3, identity);
                break ; 
          case 1:
                intersect_zsphere<true>(zrange, q0, identity);
                break ; 
          case 2:
                //intersect_aabb(q2, q3, identity);
                intersect_ztubs(zrange,q0,q1, identity);
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

/*
  rtPrintf("bounds primIdx %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, 
       result[0],
       result[1],
       result[2],
       result[3],
       result[4],
       result[5]
     );
*/

}

