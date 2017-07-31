/*
Adapting polynomial solvers from ./volumes/kernel/TorusImplementation2.h
attempting to get operational inside OptiX 
*/

#include "Solve.h"

#ifdef __CUDACC__
__device__ __host__
#endif
static unsigned SolveCubic(float a, float b, float c, float *x, unsigned msk, unsigned& path ) 
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
    const float p33 = p3*p3*p3 ;  
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
static int SolveQuartic(float a, float b, float c, float d, float *x, unsigned msk, unsigned& path  )
{
    // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    // Input: a,b,c,d
    // Output: x[4] - real solutions
    // Returns number of real solutions (0 to 3)

    const float a4 = a/4.f ; 

    float e     = b - 3.f * a * a / 8.f;
    float f     = c + a * a * a / 8.f - 0.5f * a * b;
    float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;

    //  see quartic.py : e,f,g coeff of depressed quartic (often p,q,r)
    //
    //    x^4 + a*x^3 + b*x^2 + c*x + d = 0    subs x -> z-a/4 
    //    z^4 +   0   + e z^2 + f z + g = 0 


    float xx[4] = { 1e10f, 1e10f, 1e10f, 1e10f };
    float delta;
    float h = 0.f;
    float tmp = 0.f;
    unsigned ireal = 0;

    // special case when f is zero,
    // degenerates to quadratic in z^2 : z^4 + e z^2 + g = 0 
    //   z^2 -> y 
    //                 y^2 + e y + g = 0 
   
  
    //if (fabs(f) < 1e-6f) 
    if (fabs(f) < 1e-3f)   // loosening reduces artifacts
    //if (fabs(f) < 1e-1f)     // going extreme gives visible cut-out ring 
    {
        delta = e * e - 4.f * g;
        if (delta < 0.f) return 0;

        delta = sqrt(delta);

        if( msk & SOLVE_ROBUSTCUBIC_0 )
        {  
            tmp = e > 0.f ? 0.5f*(-e - delta) : 0.5f*(-e + delta) ; 
            h = tmp ;  
        }
        else
        {
            h = 0.5f * (-e - delta);
        }


        if (h >= 0.f)
        {
            h = sqrt(h);
            x[ireal++] = -h - a4;
            x[ireal++] =  h - a4;
        }

        h = msk & SOLVE_ROBUSTCUBIC_0 ? g/tmp : 0.5f * (-e + delta) ; 

        if (h >= 0.f) 
        {
            h = sqrt(h);
            x[ireal++] = -h - a4 ;
            x[ireal++] =  h - a4 ;
        }
        return ireal;
    }


    // When g is zero : factor z, leaves depressed cubic 
    //   
    //         z^4 + e z^2 + f z + g = 0 
    //    z * (z^3 + e z   + f )  = 0       ->  z = 0  as one root
    //
    //        (z^3 + e z   + f )  = 0 
    //   

    if (fabs(g) < 1e-6f) 
    {
        x[ireal++] = -a4 ;
        // this actually wants to solve a second order equation
        // we should specialize if it happens often
        unsigned ncubicroots = SolveCubic(0, e, f, xx, msk, path );   // 0 as z**2 term already zero
        // this loop is not nice
        for (unsigned i = 0; i < ncubicroots; i++) x[ireal++] = xx[i] - a4 ;

        return ireal;
    }


    ireal = SolveCubic(2.f * e, e * e - 4.f * g, -f * f, xx, msk, path);

    //  Translations for finding resolvent cubic in literature..
    //
    //       b,c,d -> e,f,g -> p,q,r     
    //
    //      A y^4 + B y^3 + C y^2  + D y + E = 0      2*C    C^2+B*D-4*A*E    B*C*D - B^2*E - A*D^2 
    //       y**4 + 0     + b*y**2 + c*y + d = 0      2*b        b**2 - 4*d       -1 * c**2      
    //       y**4 + 0     + e*y**2 + f*y + g = 0      2*e        e**2 - 4*g       -1 * f**2
    //
    //       y**4 + 0     + p*y**2 + q*y + r = 0      2*p        p**2 - 4*r       -1 * q**2
    //
    //   So this is using Neumark resolvent, with sign flip and "a" = 0 
    //   getting fake intersects for small neumark[0] = -f*f 
    //
    //
    //   see Neumark p12, the quad solutions for g and G simplify to sqrt
    //   the below h is "x" coeff of factored quaratic

    if (ireal == 1) 
    {
        if (xx[0] <= 0.f) return 0;
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


    // hmm perhaps can use -ve h if thats numerically good ?
    // Neumark p13 with A=1, B=0 ->  0.5* ( C + h^2 - D / h )
    //
    //      j is X^0 term of one of the quadratics
    //
    //     g/j is X^0 term of other quadratic    ( p13, Hh = AE  (A=1, E=g)
    //
    //


    float j = 0.5f * (e + h * h - f / h);   
    ireal = 0;

    delta = h * h - 4.f * j;  // discrim of 1st factored quadratic


    if (delta >= 0.f) 
    {
        delta = sqrt(delta);

        if( msk & SOLVE_ROBUSTCUBIC_1 )
        { 
            tmp = h > 0.f ? 0.5f*(-h - delta) : 0.5f*(-h + delta) ; 
            x[ireal++] = tmp - a4 ;
            x[ireal++] = j/tmp - a4 ;
        }
        else
        {
            x[ireal++] = 0.5f*(-h - delta) - a4 ;
            x[ireal++] = 0.5f*(-h + delta) - a4 ; 
        }
    }


    delta = h * h - 4.f * g / j;    // discrim of 2nd factored quadratic
    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        if( msk & SOLVE_ROBUSTCUBIC_2 )
        { 
            tmp = h > 0.f ? 0.5f*(h + delta) : 0.5f*(h - delta) ; 
            x[ireal++] = tmp - a4 ;
            x[ireal++] = (g/j)/tmp - a4 ;
        }
        else
        {
            x[ireal++] = 0.5f*(h + delta) - a4 ;
            x[ireal++] = 0.5f*(h - delta) - a4 ;
        }
    }

    return ireal;
}


