/*
Adapting polynomial solvers from ./volumes/kernel/TorusImplementation2.h
attempting to get operational inside OptiX 
*/

#include "SolveEnum.h"

#ifdef __CUDACC__
__device__ __host__
#endif
static unsigned SolveCubic(float a, float b, float c, float *x, unsigned msk ) 
{
    // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
    //                                                   a2      a1    a0
    // Input: a,b,c
    // Output: x[3] real solutions
    // Returns number of real solutions (1 or 3)
    const float ott        = 1.f / 3.f; 
    const float sq3        = sqrt(3.f);
    const float inv6sq3    = 1.f / (6.f * sq3);
    unsigned int ireal = 1;

    const float p = b - a * a * ott;                                       
    const float q = c - a * b * ott + 2.f * a * a * a * ott * ott * ott;
    const float p3 = p/3.f ; 
    //const float p33 = p3*p3*p3 ;  
    const float q2 = q/2.f ; 

    //const float q22 = q2*q2 ; 
    //const float disc = p33 + q22 ;  


    float delta = 4.f * p * p * p + 27.f * q * q;   

    //  p, q are coefficients of depressed cubic :  z**3 + p*z + q = 0 
    //  obtained by substitution  x -> z - a/3  
    // 
    //  Dividing delta by 27*4 yields the cubic discriminant:   
    //
    //        delta/(27*4 ) =  (p/3)**3 + (q/2)**2         
    //
    //  sqrt of discriminant is: 
    //
    //        sqrt(delta/(3*3*3*2*2)) = sqrt(delta)/(6*sqrt(3)) = sqrt(delta)*inv6sq3
    //

    float t, u ;

    if (delta >= 0.f) // only one real root,  Cardanos formula for the depressed cubic with -a/3 shift to yield original cubic root
    {
        //path |= PATH_CUBIC_PDELTA ; 

        delta = sqrt(delta);  

        if( msk & SOLVE_VECGEOM )
        {       
            t     = (-3.f * q * sq3 + delta) * inv6sq3;  
            u     = (3.f * q * sq3 + delta) * inv6sq3;  
            x[0]  = copysign(1.f, t) * cbrt(fabs(t)) - copysign(1.f, u) * cbrt(fabs(u)) - a * ott;  

        }
        else if(msk & SOLVE_UNOBFUSCATED)
        {
            float sdisc = delta*inv6sq3 ;

            if( msk & SOLVE_ROBUST_VIETA )
            {
                t = q2 < 0.f ? -q2 + sdisc : q2 + sdisc ;  
            
                float tcu = copysign(1.f, t) * cbrt(fabs(t)) ; 
                float ucu = p3 / tcu ;        
               
                //  Evaluate the more numerically advantageous of the below branches t and u
                //  and then get the other using Vieta subs:    
                //
                //        x = w - p3/w    (where w corresponds to t or u )
                //
                //  product of cube roots in -p/3 ( see Classical Algebra, Cooke, p66 )     
                //
                //  THIS TRICK YIELDS A LOT LESS ARTIFACTS FROM NUMERICAL IMPRECISION
            
                x[0]  = q2 < 0.f ? tcu - ucu : ucu - tcu ;
            }
            else
            {
                t = -q2 + sdisc ;
                u =  q2 + sdisc ;

                float tcu = copysign(1.f, t) * cbrt(fabs(t)) ;
                float ucu = copysign(1.f, u) * cbrt(fabs(u)) ; 
      
                x[0]  = tcu - ucu ;  
            }
            x[0]  -=  a * ott;  
        }
    } 
    else 
    {
        //path |= PATH_CUBIC_NDELTA ; 

        delta = sqrt(-delta);
        t     = -0.5f * q;
        u     = delta * inv6sq3;      // sqrt of negated discrim :  sqrt( -[(p/3)**3 + (q/2)**2] )

        // potentially should pick based on values for better numerical handling ?
        if( msk & SOLVE_VECGEOM )
        {       
            x[0]  = 2.f * pow(t * t + u * u, 0.5f * ott) * cos(ott * atan2(u, t));
        }
        else if(msk & SOLVE_UNOBFUSCATED)
        {
            x[0]  = 2.f * sqrt(-p/3.f) * cos(ott * atan2(u, t)); 
        }
        
        //  obfuscation ???
        //
        //       pow( t**2 + u**2, 0.5/3 ) 
        //       pow( (q/2)**2 -[(p/3)**3 + (q/2)**2] , 0.5/3 ) 
        //       pow(  (-p/3)**3 , 0.5/3 )
        //       pow(  (-p/3) , 0.5 )
        //       sqrt( -p/3 )         ## NB this branch implies p<0
        //
        //  Viete substitution (see eg Classical Algrebra, Roger Cooke, p78
        //  Trisection identity motivates angle
        //
        //                    
        //                    3*sqrt(3)
        //       cos(phi) = -----------  * (-q/2)  =   sqrt( 27/(-p)**3 )* (-q/2)
        //                   -p*sqrt(-p)
        //
        //       sin(phi) = sqrt( (q/2)**2 + (p/3)**3 ) * sqrt( 27/p**3 )
        //
        //        y = 2*sqrt(-p/3)*cos(1/3  * arccos(q/2* 3*sqrt(3)/p*sqrt(p)) ) 
        //
        //        

        x[0] -= a * ott;
    }


    // polynomial long division 
    //     x**3 + a x**2 + b x + x = 0 
    //  (x-x0)( x**2 + (x+x0)x + b + a*x0 + x0**2 )   remainder is one degree less with x->x0
    //     

    t     = x[0] * x[0] + a * x[0] + b;
    u     = a + x[0];
    delta = u * u - 4.f * t;

    if (delta >= 0.f) 
    {
        ireal = 3;
        delta = sqrt(delta);
   
        if( msk & SOLVE_ROBUSTQUAD_1 )
        {
            float tmp = u > 0.f ? 0.5f*(-u - delta) : 0.5f*(-u + delta) ; 
            x[1] = tmp ; 
            x[2] = t/tmp ; 
        }
        else
        {
            x[1]  = 0.5f * (-u - delta);
            x[2]  = 0.5f * (-u + delta);
        }

    }
    return ireal;
}







#ifdef __CUDACC__
__device__ __host__
#endif
static int qudrtc(float b, float c, float *rts, float disc, float offset )
{
/* 
     solve the quadratic equation :  x**2+b*x+c = 0 
        c=0 ->   x(x+b) = x**2 + b*x = 0  -> x=0, x=-b

*/
    float inv2 = 0.5f ;
    int nreal = 0 ;
    if(disc >= 0.f)
    {
        float sdisc = sqrt(disc) ;
        nreal = 2 ;
        rts[0] = b > 0.f ? -inv2*( b + sdisc) : -inv2*( b - sdisc)  ;
        rts[1] = rts[0] == 0.f ? -b : c/rts[0] ;
        rts[0] += offset ; 
        rts[1] += offset ; 
    }
    return nreal ;
}



#ifdef __CUDACC__
__device__ __host__
#endif
static float cubic_real_root(float p, float q, float r, unsigned msk)
{
    float xx[3] ; 
    unsigned ireal = SolveCubic(p, q, r, xx, msk);
    float h = 0.f ; 

    if (ireal == 1) 
    {
        if (xx[0] <= 0.f) return 0 ;
        h = sqrt(xx[0]);
    } 
    else 
    {
        // 3 real solutions of the cubic
        for (unsigned i = 0; i < 3; i++) 
        {
            h = xx[i];
            if (h >= 0.f) break;
        }
        if (h <= 0.f) return 0;
        h = sqrt(h);
    }
    return h ; 
}




#ifdef __CUDACC__
__device__ __host__
#endif
static float cubic_lowest_real_root(float p, float q, float r)
{
/* 
     find the lowest real root of the cubic - 
       x**3 + p*x**2 + q*x + r = 0 

   input parameters - 
     p,q,r - coeffs of cubic equation. 

   output- 
     cubic - a real root. 

   global constants -
     rt3 - sqrt(3) 
     inv3 - 1/3 
     doubmax - square root of largest number held by machine 

     method - 
     see D.E. Littlewood, "A University Algebra" pp.173 - 6 

     Charles Prineas   April 1981 

     called by  neumark.
     calls  acos3 
*/

   float rt3 = sqrt(3.f);
   float doub1 = 1.f ; 
   float inv2 = 1.f/2.f ; 
   float inv3 = 1.f/3.f ; 
   float nought = 0.f ; 

   int nrts = 0 ;
   float po3,po3sq,qo3;
   float uo3,u2o3,uo3sq4,uo3cu4 ;
   float v,vsq,wsq ;
   float m,mcube,n;
   float muo3,s,scube,t,cosk,sinsqk ;
   float root;

//   float curoot();
//   float acos3();
//   float sqrt(),fabs();

   m = nought;

/*
   if ( p > doubmax || p <  -doubmax)  // x**3 + p x**2 + q *x + r = 0 ->   x + p = 0   (p-dominant)
   {
       root = -p;
   }
   else if ( q > doubmax || q <  -doubmax )  //   x**2 = -q   ???  
   {
       root = q > nought ? -r/q : -sqrt(-q) ;
   }
   else if ( r > doubmax ||  r <  -doubmax ) //  x**3 = -r 
   {
       root =  -curoot(r) ;
   }
   else
*/
   {
       po3 = p*inv3 ;
       po3sq = po3*po3 ;

/*
       if (po3sq > doubmax) 
       {
           root =  -p ;
       }
       else
*/
       {
           v = r + po3*(po3sq + po3sq - q) ;

/*
           if ((v > doubmax) || (v < -doubmax)) 
           {
               root = -p ;
           }
           else
*/
           {
               vsq = v*v ;
               qo3 = q*inv3 ;
               uo3 = qo3 - po3sq ;
               u2o3 = uo3 + uo3 ;

/*
               if ((u2o3 > doubmax) || (u2o3 < -doubmax))
               {
                   root = p == nought ? ( q > nought ? -r/q : -sqrt(-q)  ) : -q/p ; 
               }
*/
              
               uo3sq4 = u2o3*u2o3 ;

/*
               if (uo3sq4 > doubmax)
               {
                   root = p == nought ? ( q > nought ? -r/q : -sqrt(fabs(q))  ) : -q/p ; 
               }
*/

               uo3cu4 = uo3sq4*uo3 ;
               wsq = uo3cu4 + vsq ;
               if (wsq >= nought)
               {
                   nrts = 1;  // cubic has one real root 
                   mcube = v <= nought ? ( -v + sqrt(wsq))*inv2 : ( -v - sqrt(wsq))*inv2 ;

                   m = cbrtf(mcube) ;
                   n = m != nought ? -uo3/m : nought ;

                   root = m + n - po3 ;
               }
               else
               {
                   nrts = 3;  // cubic has three real roots 

                   if (uo3 < nought)
                   {
                       muo3 = -uo3;
                       s = sqrt(muo3) ;
                       scube = s*muo3;
                       t =  -v/(scube+scube) ;
                       
                       //cosk = acos3(t) ;
                       cosk = cos(acos(t)*inv3) ;

                       if (po3 < nought)
                       {
                           root = (s+s)*cosk - po3;
                       }
                       else
                       {
                           sinsqk = doub1 - cosk*cosk ;
                           if (sinsqk < nought) sinsqk = nought ;
                           root = s*( -cosk - rt3*sqrt(sinsqk)) - po3 ;
                       }
                   }
                   else
                   {
                       root = cbrtf(v) - po3 ;
                   }
               }
           }
       }
   }
   return root ;
} 






/*  
    see quartic.py : e,f,g coeff of depressed quartic (often p,q,r)
    
        x^4 + a*x^3 + b*x^2 + c*x + d = 0    subs x -> z-a/4 
        z^4 +   0   + e z^2 + f z + g = 0 


    from sympy import symbols, expand, Poly
    a,b,c,d,x,y,z = symbols("a:d,x:z")
    ex = x**4 + a*x**3 + b*x**2 + c*x + d 
    ey = expand(ex.subs(x,y-a/4))
    cy = Poly(ey,y).all_coeffs()   

    cy coeffs of depressed quartic:: 

         [1, 
          0, 
         -3*a**2/8 + b, 
          a**3/8 - a*b/2 + c, 
          -3*a**4/256 + a**2*b/16 - a*c/4 + d]

       e,f,g = cy[2:]

       In [54]: map(expand, [2*e, e*e - 4* g, -f*f ])
        Out[54]: 
        [-3*a**2/4 + 2*b,
         3*a**4/16 - a**2*b + a*c + b**2 - 4*d,
         -a**6/64 + a**4*b/8 - a**3*c/4 - a**2*b**2/4 + a*b*c - c**2]


     Resolvent cubic is Descartes-Euler-Cardano 
     which involves many high powers, which will be terrible numerically...


    Suspect the difference is that Cardano first substitutes to 
    depress the quartic, whereas Neumark directly 
    factors the quartic.  Depressing the quartic results in 
    lots of high powers

*/


#ifdef __CUDACC__
__device__ __host__
#endif
static int SolveQuartic(float a, float b, float c, float d, float *x, unsigned msk  )
{
    // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    // Input: a,b,c,d
    // Output: x[4] - real solutions
    // Returns number of real solutions (0 to 3)

    const float a4 = a/4.f ; 

    // (1,0,e,f,g) are coeff of depressed quartic
    const float e     = b - 3.f * a * a / 8.f;
    const float f     = c + a * a * a / 8.f - 0.5f * a * b;
    const float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;

    //  (1,p,q,r) are coeffs of resolvent cubic
    const float p = 2.f * e ;
    const float q = e * e - 4.f * g ;
    const float r = -f * f ;

#ifdef SOLVE_QUARTIC_DEBUG
    rtPrintf("SolveQuartic"
             " abcd (%g %g %g %g) " 
             " pqr (%g %g %g) " 
             "\n"
             ,
             a,b,c,d
             ,
             p,q,r
            );
#endif

    float xx[4] = { 1e10f, 1e10f, 1e10f, 1e10f };
    float delta;
    float h = 0.f;
    unsigned ireal = 0;

    // special case when f is zero,
    // degenerates to quadratic in z^2 : z^4 + e z^2 + g = 0 
    //   z^2 -> y 
    //                 y^2 + e y + g = 0 
    // 

    //if (fabs(f) < 1e-6f) 
    //if (fabs(f) < 1e-3f)   // loosening reduces artifacts
    //if (fabs(r) < 1e-3f)   // loosening reduces artifacts
    //if (fabs(f) < 1e-1f)   // going extreme gives visible cut-out ring 

    // apply directly to r removes artifacts and seems not to cut 
    // but this only works for small torus with major radii 1.
    //

    if (fabs(r) < 1e-3f)     
    {
        delta = e * e - 4.f * g;
 
        float quad[2] ; 
        int iquad = qudrtc( e,  g , quad, delta, 0.f ) ; 
        for(int i=0 ; i < iquad ; i++)
        {
            h = quad[i] ;
            if(h >= 0.f)
            { 
                h = sqrt(h);
                x[ireal++] = -h - a4 ; 
                x[ireal++] =  h - a4 ; 
            }
        } 
        return ireal;
    }


    // When g is zero : factor z, leaves depressed cubic 
    //   
    //         z^4 + e z^2 + f z + g = 0 
    //    z * (z^3 + e z  + f )  = 0       ->  z = 0  root

    if (fabs(g) < 1e-6f)   
    {
        x[ireal++] = -a4 ;   // z=0 root

        unsigned ncubicroots = SolveCubic(0, e, f, xx, msk );   // 0 as z**2 term already zero

        for (unsigned i = 0; i < ncubicroots; i++) x[ireal++] = xx[i] - a4 ;

        return ireal;
    }

    // when r is small < -1e-3
    // a small cubic root is returned by the below with 
    // a very large error > 25%  (from poly residual/deriv) 
    // ... perhaps better to use another real root if there is one
    // lack of const term means a zero root,
    // but h=0 causes problem for j below ?
    // 
    // Neumark footnote on p14 has alt expression for just this case
    //     
    //
    // changing from an f cut to an r cut above
    // succeeds to eliminate artifacts without visible cutting
    // BUT enlarging up from major radius 1. to 300. brings back the artifacts
    //
    // Attempting to use scaling producing wierd shape... scaling makes ray_direction
    // real small to support.
    // Perhaps move to normalized ray_direction for torus along, dropping non-uni support
    // for torus. 
    //

    //h = cubic_lowest_real_root( p, q, r );
    h = cubic_real_root( p, q, r, msk );  // sqrt of cubic root

    //if (h < 0.001f) return 0;  // hairline crack

    if (h <= 0.f) return 0;

  
    float j = 0.5f * (e + h * h - f / h);   

    ireal = 0;

    float dis1 = h * h - 4.f * j;  // discrim of 1st factored quadratic

    ireal += qudrtc( h, j , x, dis1, -a4 ) ; 

    float dis2 = h * h - 4.f * g / j;    // discrim of 2nd factored quadratic

    ireal += qudrtc( -h, g/j , x+ireal, dis2, -a4 ) ; 


    for(int i=0 ; i < ireal ; i++)
    {
        float residual = (((x[i] + a)*x[i] + b)*x[i] + c)*x[i] + d ; 

        if(residual > 100.f )
        {
           rtPrintf(
                 " ireal %d i %d root %g residual %g "
                 " dis12 ( %g %g ) h %g "
                 " pqr (%g %g %g ) "  
                 " j g/j (%g %g ) "
                 " \n"
                 ,
                 ireal
                 ,
                 i
                 , 
                 x[i]
                 ,
                 residual
                 ,
                 dis1, dis2, h
                 ,
                 p,q,r
                 ,
                 j, g/j
                 
            );
                 
        }

    }
    return ireal;
}


/*
                 " efg (%g %g %g ) "  
                 e,f,g


                 " abcd (%g %g %g %g ) "  
                 a,b,c,d
                 ,


*/







#ifdef __CUDACC__
__device__ __host__
#endif
static int SolveQuarticPureNeumark(float a, float b, float c, float d, float* rts, unsigned msk  )
{
   //  Neumark p12
    //
    //  Ax**4 + B*x**3 + C*x**2 + D*x + E = 0
    //   x**4 + a*x**3 + b*x**2 + c*x + d = 0 
    //
    //            y**3 + p*y**2 + q*y + r = 0     resolvent cubic
    //
    //
    //    -2*C
    //    -2*b
    //
    //    C*C + B*D - 4*A*E
    //    b*b + a*c - 4*1*d
    //
    //    -B*C*D + B*B*E + A*D*D
    //    -a*b*c + a*a*d + 1*c*c 

    // cf /usr/local/env/geometry/quartic/quartic/quarcube.c
    //   http://www.realtimerendering.com/resources/GraphicsGems/gemsv/ch1-1/quarcube.c
    //

    const float nought = 0.f ; 
    const float two = 2.f ; 
    const float four = 4.f ; 
    const float inv2 = 0.5f ; 

    const float asq = a*a ; 

    const float p = -two*b ; 
    const float q = b*b + a*c - four*d ; 
    const float r = (c-a*b)*c + asq*d  ;  

#ifdef PURE_NEUMARK_DEBUG
    rtPrintf(" PURE_NEUMARK_DEBUG " 
             " abcd (%g %g %g %g) "
             " pqr ( %g %g %g ) "
             ,
             a,b,c,d
             ,
             p,q,r
            );
#endif
     
    if (fabs(r) < 1e-3f)     
    {
        // Neumark p18 : constant term of (3.9) is zero -> resolvent cubic has one zero root
        // giving immediate factorization
        //    
        //     (A x**2 + B x + ( C - A*D/B )) * ( x**2 + D/B )  = 0 
        //     (  x**2 + a x + ( b - c/a ) ) * ( x**2 + c/a ) = 0     
 
        int ireal = 0 ; 
 
        float coa = c/a ; // perfect sq term   
        if(coa >= 0.f )
        {
            float scoa = sqrt(coa) ; 
            rts[ireal++] = scoa ; 
            rts[ireal++] = -scoa ; 
        }
             
        float rdis = a * a - 4.f*(b-coa) ;
        if(rdis > 0.f)
        {
            float quad[2] ; 
            qudrtc( a, b-coa, quad, rdis, 0.f ) ;
            rts[ireal++] = quad[0] ; 
            rts[ireal++] = quad[1] ; 
        } 

#ifdef PURE_NEUMARK_DEBUG
        rtPrintf(" PURE_NEUMARK_DEBUG small-r " 
                 " r coa rdis ireal (%g %g %g ; %d ) "
                 ,
                 r,coa,rdis,ireal
                 ); 
#endif

        return ireal;
    }



    //float y = cubic_real_root( p, q, r, msk );
    float y = cubic_lowest_real_root( p, q, r );

    //y = 25.f ; 
    if (y <= 0.f) return 0 ;


    const float bmy = b - y ;
    const float y4 = y*four ; 
    const float d4 = d*four ;
    const float bmysq = bmy*bmy ;
    const float gdis = asq - y4 ;
    const float hdis = bmysq - d4 ;
    if ( gdis < nought || hdis < nought ) return 0 ;

    // see Graphics Gems : Solving Quartics and Cubics for Graphics, Herbison-Evans

    const float g1 = a*inv2 ;
    const float h1 = bmy*inv2 ;
    const float gerr = asq + y4 ;        // p9: asq - 4y  ??
    const float herr = d > nought ? bmysq + d4 : hdis ; 

    float g2 ; 
    float h2 ; 

    if ( y < nought || herr*gdis > gerr*hdis )
    {
        const float gdisrt = sqrt(gdis) ;
        g2 = gdisrt*inv2 ;
        h2 = gdisrt != nought ? (a*h1 - c)/gdisrt : nought ;
    }
    else
    {
        const float hdisrt = sqrt(hdis) ;
        h2 = hdisrt*inv2 ;
        g2 = hdisrt != nought ? (a*h1 - c)/hdisrt : nought ;
    }


#ifdef PURE_NEUMARK_DEBUG

    rtPrintf(" PURE_NEUMARK_DEBUG " 
             " abcd (%g %g %g %g) "
             " pqr ( %g %g %g ) "
             " y bmy (%g %g ) "
             " gdis hdis (%g %g ) "
             " g1 h1 g2 h2 (%g %g %g %g) "
             " gerr herr (%g %g ) "
             ,
             a,b,c,d
             ,
             p,q,r
             ,
             y,bmy
             ,
             gdis, hdis
             ,
             g1, h1, g2, h2
             ,
             gerr, herr
             );

#endif


    // note that in the following, the tests ensure non-zero denominators -  

    float h ;
    float hh ;
    float hmax ; 

    h = h1 - h2 ;
    hh = h1 + h2 ;

    hmax = hh ;
    if (hmax < nought) hmax =  -hmax ;
    if (hmax < h) hmax = h ;
    if (hmax <  -h) hmax =  -h ;

    if ( h1 > nought && h2 > nought ) h = d/hh ;
    if ( h1 < nought && h2 < nought ) h = d/hh ;
    if ( h1 > nought && h2 < nought ) hh = d/h ;
    if ( h1 < nought && h2 > nought ) hh = d/h ;
    
    if ( h > hmax) h = hmax ;
    if ( h <  -hmax) h =  -hmax ;
    if ( hh > hmax) hh = hmax ;
    if ( hh <  -hmax) hh =  -hmax ;


    float g ; 
    float gg ; 
    float gmax ; 

    g = g1 - g2 ;
    gg = g1 + g2 ;
    gmax = gg ;

    if (gmax < nought) gmax =  -gmax ;
    if (gmax < g) gmax = g ;
    if (gmax <  -g) gmax =  -g ;

    if ( g1 > nought && g2 > nought ) g = y/gg ;
    if ( g1 < nought && g2 < nought ) g = y/gg ;
    if ( g1 > nought && g2 < nought ) gg = y/g ;
    if ( g1 < nought && g2 > nought ) gg = y/g ;

    if (g > gmax) g = gmax ;
    if (g <  -gmax) g =  -gmax ;
    if (gg > gmax) gg = gmax ;
    if (gg <  -gmax) gg =  -gmax ;


    float v1[2],v2[2] ;

    float disc1 = gg*gg - four*hh ;
    float disc2 =  g*g - four*h ;

    int n1 = qudrtc(gg, hh,v1, disc1, nought ) ;
    int n2 = qudrtc( g,  h,v2, disc2, nought ) ;

    int nquar = n1+n2 ;



#ifdef PURE_NEUMARK_DEBUG

    rtPrintf(" PURE_NEUMARK_DEBUG " 
             " gg hh disc1 n1 (%g %g %g ; %d) "
             " v1[0] v1[1] ( %g %g ) "
             " g h disc2 n2 (%g %g %g ; %d) "
             " v2[0] v2[1] ( %g %g ) "
             "\n"
             ,
             gg,hh,disc1,n1 
             ,
             v1[0],v1[1]
             ,
             g,h,disc2,n2
             ,
             v2[0],v2[1]
             )
             ;  
#endif




    rts[0] = v1[0] ;
    rts[1] = v1[1] ;
    rts[n1+0] = v2[0] ;
    rts[n1+1] = v2[1] ;
 

    return nquar ; 


}





//  Translations for finding resolvent cubic in literature..
//
//       b,c,d -> e,f,g -> p,q,r     
//
//      A y^4 + B y^3 + C y^2  + D y + E = 0      2*C    C^2+B*D-4*A*E    B*C*D - B^2*E - A*D^2 
//       y**4 + a*y**3 + b*y**2 + c*y + d = 0      2*b        b**2 - 4*d       -1 * c**2      
//       y**4 + 0      + e*y**2 + f*y + g = 0      2*e        e**2 - 4*g       -1 * f**2
//
//       y**4 + 0     + p*y**2 + q*y + r = 0      2*p        p**2 - 4*r       -1 * q**2
//
//   So this is using Neumark resolvent, with sign flip and "a" = 0 
//   getting fake intersects for small neumark[0] = -f*f 
//
//
//   see Neumark p12, the quad solutions for g and G simplify to sqrt
//   the below h is "x" coeff of factored quaratic
//
// Neumark p12 (3.1)
//
//    A y**4 + B y**3 + C y**2 + D y + E = 0 
//    1 y**4 + a y**3 + b y**2 + c y + d = 0 
//    1 y**4 + 0 y**3 + e y**2 + f y + g = 0 
//
//
// Neumark p13 (3.8b) 
//
//    ( A*x**2 + G*x + H )*(A*x**2 + g*x + h ) = 0 
//
//   A->1 
//    discrim 
//          G**2 - 4*H  ,      g**2 - 4*h    
//
//
//           B + sqrt(B*B - 4*x)
//     G =   ---------------------- 
//                  2
//           B - sqrt(B*B - 4*x)
//     g =   ---------------------- 
//                  2
//
//
//           a + sqrt(a*a - 4*y)
//     G =   ---------------------- 
//                  2
//           a - sqrt(a*a - 4*y)
//     g =   ---------------------- 
//                  2
//
//     G + g = B  = a   
//     G*g   = Ax = y         (a+sq)/2(a-sq)/2 = (a^2 - a^2 + 4y)/4 = y 
//
//
//
//           C - x      B(C - x) - 2*A*D
//     H =  ------- +  ------------------
//             2        2*sqrt(B*B - 4*A*x)
//
//           C - x      B(C - x) - 2*A*D
//     h =  ------- -  ------------------
//             2        2*sqrt(B*B - 4*A*x)
//
//   
//
//           b - y      a*(b - y) - 2*c
//     H =  ------- +  ------------------
//             2        2*sqrt(a*a - 4*y)
//
//           b - y      a*(b - y) - 2*c
//     h =  ------- -  ------------------  
//             2        2*sqrt(a*a - 4*y)
//
//
//     G*G - 4*H  
//
//         a^2 + 2*sqrt(a*a - 4y) + (a*a - 4*y)
//        ---------------------------------------  -    
//                        4 
//        
//
//
//
//
//     h*H = A*E    
//  -> h*H = g
//
//     A = 1, B = 0 
//
//           C - x        - D
//     H =  ------- +  ----------
//             2         2*sqrt(-x)
//    
//        = 0.5*( e - x - f/sqrt(-x) )
//        = 0.5*( e + h^2 - f/h  )        looks like below j with   h = sqrt(-x) 
// 
//
//      j is X^0 term of one of the quadratics
//
//     g/j is X^0 term of other quadratic    ( p13, Hh = AE  (A=1, E=g)
//
//

