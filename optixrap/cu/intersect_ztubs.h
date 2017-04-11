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

    // see opticks/ana/pmt/geom.py:Part.as_quads
    float sizeZ = q1.f.x ; 
    float z0 = q0.f.z - sizeZ/2.f ;     
    float3 position = make_float3( q0.f.x, q0.f.y, z0 );  // 0,0,-169.

    float zmin = q2.f.z ;
    float zmax = q3.f.z ;

    float clipped_sizeZ = zmax - zmin ;  
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


