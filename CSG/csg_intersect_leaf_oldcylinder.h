#pragma once
/**
intersect_leaf_oldcylinder
------------------------

For ascii art explanation of the maths see optixrap/cu/intersect_ztubs.h

* handling inner radius within the primitive would be useful, but need to simplify first 
* ideas to simplify

  * adopt natural cylinder frame, require abs(z1) = abs(z2) ie -z:z 
  * split into separate methods for infinite intersect 


**/

#define CSG_OLDCYLINDER_PRECISION_ISSUE 1 

LEAF_FUNC
bool intersect_leaf_oldcylinder( float4& isect, const quad& q0, const quad& q1, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    const float   radius = q0.f.w ; 
    const float       z1 = q1.f.x  ; 
    const float       z2 = q1.f.y  ; 

    const float  sizeZ = z2 - z1 ; 
    const float3 position = make_float3( q0.f.x, q0.f.y, z1 ); // P: point on axis at base of cylinder

#ifdef DEBUG_RECORD
    printf("//[intersect_leaf_cylinder radius %10.4f z1 %10.4f z2 %10.4f sizeZ %10.4f \n", radius, z1, z2, sizeZ ); 
#endif
    const float3 m = ray_origin - position ;          // m: ray origin in cylinder frame (cylinder origin at base point P)
    const float3 n = ray_direction ;                  // n: ray direction vector (not normalized)
    const float3 d = make_float3(0.f, 0.f, sizeZ );   // d: (PQ) cylinder axis vector (not normalized)

    float rr = radius*radius ; 
    float3 dnorm = normalize(d);  

    float mm = dot(m, m) ;   //    
    float nn = dot(n, n) ;   // all 1. when ray_direction normalized
    float dd = dot(d, d) ;   // sizeZ*sizeZ 
    float nd = dot(n, d) ;   // dot product of axis vector and ray, ie -0.3/0.3 (for hz 0.15 cyl)
    float md = dot(m, d) ;
    float mn = dot(m, n) ; 
    float k = mm - rr ; 

    // quadratic coefficients of t,     a tt + 2b t + c = 0 
    // see bk-;bk-rtcdcy for derivation of these coefficients

    float a = dd*nn - nd*nd ;   
    float b = dd*mn - nd*md ;
    float c = dd*k - md*md ; 


    float disc = b*b-a*c;
    float t_cand = t_min ; 


    enum {  ENDCAP_P=1,  ENDCAP_Q=2 } ; 

    // axial ray endcap handling : can treat axial rays in 2d way 
    if(fabs(a) < 1e-6f)     
    {

#ifdef DEBUG_RECORD
    printf("//intersect_leaf_cylinder : axial ray endcap handling, a %10.4g c(dd*k - md*md) %10.4g dd %10.4g k %10.4g md %10.4g  \n", a, c,dd,k,md ); 
#endif
        if(c > 0.f) return false ;  // ray starts and ends outside cylinder


#ifdef CSG_OLDCYLINDER_PRECISION_ISSUE
        float t_PCAP_AX = -mn/nn  ;      
        float t_QCAP_AX = (nd - mn)/nn ;  
    
        if(md < 0.f )     // ray origin on P side  
        {
            t_cand = t_PCAP_AX > t_min ? t_PCAP_AX : t_QCAP_AX ;
        } 
        else if(md > dd )  // ray origin on Q side 
        {
            t_cand = t_QCAP_AX > t_min ? t_QCAP_AX : t_PCAP_AX ;
        }
        else              // ray origin inside,   nd > 0 ray along +d towards Q  
        {
            t_cand = nd > 0.f ? t_QCAP_AX : t_PCAP_AX ;  
        }


#else
        float t_PCAP_AX = copysignf(1.f, ray_direction.z)*(z1 - ray_origin.z) ;  // up/down oriented to match the dot product approach but more simply
        float t_QCAP_AX = copysignf(1.f, ray_direction.z)*(z2 - ray_origin.z) ;  // up/down oriented to match the dot product approach but more simply

        if(ray_origin.z < z1)
        {
            t_cand = t_PCAP_AX > t_min ? t_PCAP_AX : t_QCAP_AX ;
        }
        else if( ray_origin.z > z2 )
        {
            t_cand = t_QCAP_AX > t_min ? t_QCAP_AX : t_PCAP_AX ;
        }
        else              // ray origin inside,   nd > 0 ray along +d towards Q  
        {
            t_cand = ray_direction.z > 0.f ? t_QCAP_AX : t_PCAP_AX ;  
        }
#endif

        unsigned endcap = t_cand == t_PCAP_AX ? ENDCAP_P : ( t_cand == t_QCAP_AX ? ENDCAP_Q : 0u ) ;

        bool has_axial_intersect = t_cand > t_min && endcap > 0u ;

        if(has_axial_intersect)
        {
            float sign = endcap == ENDCAP_P ? -1.f : 1.f ;  
            isect.x = sign*dnorm.x ; 
            isect.y = sign*dnorm.y ; 
            isect.z = sign*dnorm.z ; 
            isect.w = t_cand ;      

#ifdef DEBUG_CYLINDER
            CSGDebug_Cylinder dbg = {} ; 

            dbg.ray_origin = ray_origin ;   // 0
            dbg.rr = rr ; 

            dbg.ray_direction = ray_direction ;  // 1 
            dbg.k  = k ; 

            dbg.m = m ;     // 2
            dbg.mm = mm ;  

            dbg.n = n ;     // 3
            dbg.nn = nn ;  

            dbg.d = d ;     // 4
            dbg.dd = dd ;  

            dbg.nd = nd ;   // 5 
            dbg.md = md ; 
            dbg.mn = mn ; 
            dbg.checkz = ray_origin.z+ray_direction.z*t_cand ;

            dbg.a = a ;    // 6 
            dbg.b = b ; 
            dbg.c = c ; 
            dbg.disc = disc ; 

            dbg.isect = isect ;      // 7 

            CSGDebug_Cylinder::record.push_back(dbg); 
#endif
        }

        return has_axial_intersect ;
    }   // end-of-axial-ray endcap handling 
    


    if(disc > 0.0f)  // has intersections with the infinite cylinder
    {
        float t_NEAR, t_FAR, sdisc ;   
        robust_quadratic_roots(t_NEAR, t_FAR, disc, sdisc, a, b, c); //  Solving:  a t^2 + 2 b t +  c = 0 

#ifdef DEBUG_CYLINDER
        /*
        // see CSG/sympy_cylinder.py 
        const float& ox = ray_origin.x ; 
        const float& oy = ray_origin.y ; 
        const float& vx = ray_direction.x ; 
        const float& vy = ray_direction.y ; 

        float a1 = vx*vx + vy*vy ; 
        float b1 = ox*vx + oy*vy ; 
        float c1 = ox*ox + oy*oy - rr ; 

        float disc1 = b1*b1-a1*c1;
        
        //printf("// intersect_leaf_cylinder  a %10.4f a1 %10.4f a/a1 %10.4f  b %10.4f b1 %10.4f b/b1 %10.4f     c %10.4f c1 %10.4f c/c1 %10.4f  \n", 
        //    a, a1, a/a1, 
        //    b, b1, b/b1, 
        //    c, c1, c/c1 ); 
   

        float t_NEAR1, t_FAR1, sdisc1 ;   
        robust_quadratic_roots(t_NEAR1, t_FAR1, disc1, sdisc1, a1, b1, c1); //  Solving:  a t^2 + 2 b t +  c = 0 

        printf("// intersect_leaf_cylinder  t_NEAR %10.4f t_NEAR1 %10.4f t_FAR %10.4f t_FAR1 %10.4f \n", t_NEAR, t_NEAR1, t_FAR, t_FAR1 );  

        */
#endif


        float t_PCAP = -md/nd ; 
        float t_QCAP = (dd-md)/nd ;   


        float aNEAR = md + t_NEAR*nd ;        // axial coord of near intersection point * sizeZ
        float aFAR  = md + t_FAR*nd ;         // axial coord of far intersection point  * sizeZ

        float3 P1 = ray_origin + t_NEAR*ray_direction ;  
        float3 P2 = ray_origin + t_FAR*ray_direction ;  

        float3 N1  = (P1 - position)/radius  ;     // HMM: subtracting fixed position at base ?
        float3 N2  = (P2 - position)/radius  ;     // that is wrong for the z component, but z component is zero so no problem 

        float checkr = 0.f ; 
        float checkr_PCAP = k + t_PCAP*(2.f*mn + t_PCAP*nn) ; // bracket typo in RTCD book, 2*t*t makes no sense   
        float checkr_QCAP = k + dd - 2.0f*md + t_QCAP*(2.f*(mn-nd)+t_QCAP*nn) ;             


        if( aNEAR > 0.f && aNEAR < dd )  // near intersection inside cylinder z range
        {
            t_cand = t_NEAR ; 
            checkr = -1.f ; 
        } 
        else if( aNEAR < 0.f ) //  near intersection outside cylinder z range, on P side
        {
            t_cand =  nd > 0.f ? t_PCAP : t_min ;   // nd > 0, ray headed upwards (+z)
            checkr = checkr_PCAP ; 
        } 
        else if( aNEAR > dd ) //  intersection outside cylinder z range, on Q side
        {
            t_cand = nd < 0.f ? t_QCAP : t_min ;  // nd < 0, ray headed downwards (-z) 
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
        else if( aFAR < 0.f ) //  far intersection outside cylinder z range, on P side (-z)
        {
            t_cand = nd < 0.f ? t_PCAP : t_min ;      // sign flip cf RTCD:p198     
            checkr = checkr_PCAP ; 
        } 
        else if( aFAR > dd ) //  far intersection outside cylinder z range, on Q side (+z)
        {
            t_cand = nd > 0.f ? t_QCAP : t_min  ;    // sign flip cf RTCD:p198
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



