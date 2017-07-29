/*
Adapting polynomial solvers from ./volumes/kernel/TorusImplementation2.h
attempting to get operational inside OptiX 
*/

#include "Solve.h"

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

    float p                = b - a * a * ott;                                       
    float q                = c - a * b * ott + 2.f * a * a * a * ott * ott * ott;
    //
    //  p, q are coefficients of depressed cubic :  z**3 + p*z + q = 0 
    //  obtained by substitution  x -> z - a/3  
    //
  
    float delta            = 4.f * p * p * p + 27.f * q * q;   

    float ppp27 =  p*p*p/27.f ;  


    //  Dividing delta by 27*4 yields the cubic discriminant:   
    //
    //        delta/(27*4 ) =  (p/3)**3 + (q/2)**2         
    //
    //  sqrt of discriminant is: 
    //
    //        sqrt(delta/(3*3*3*2*2)) = sqrt(delta)/(6*sqrt(3)) = sqrt(delta)*inv6sq3
    //

    float t, u;

    if (delta >= 0.f) // only one real root,  
    {
        delta = sqrt(delta);  
        if( msk & SOLVE_VECGEOM )
        {       
            t     = (-3.f * q * sq3 + delta) * inv6sq3;  
            u     = (3.f * q * sq3 + delta) * inv6sq3;  
        }
        else if(msk & SOLVE_UNOBFUSCATED)
        {
            float sdisc = delta*inv6sq3 ;
            if( msk & SOLVE_ROBUSTQUAD )
            {
                t = q < 0.f ? -q/2.f + sdisc : ppp27/(q/2.f + sdisc)  ; 
                u = t + q  ;
            }
            else
            {
                t = -q/2.f + sdisc ;
                u =  q/2.f + sdisc ;
            }
        }
        // Cardanos formula for the depressed cubic with -a/3 shift to yield original cubic root
        x[0]  = copysign(1.f, t) * cbrt(fabs(t)) - copysign(1.f, u) * cbrt(fabs(u)) - a * ott;  
 
    } 
    else 
    {
        delta = sqrt(-delta);
        t     = -0.5f * q;
        u     = delta * inv6sq3;      // sqrt of negated discrim :  sqrt( -[(p/3)**3 + (q/2)**2] )

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
   
        if( msk & SOLVE_ROBUSTQUAD )
        {
            float tmp = u > 0.f ? 0.5f*(-u - delta) : 0.5f*(-u + delta) ; 
            x[1] = tmp/1.f ; 
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
static int SolveQuartic(float a, float b, float c, float d, float *x, unsigned msk  )
{
    // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    // Input: a,b,c,d
    // Output: x[4] - real solutions
    // Returns number of real solutions (0 to 3)

    float e     = b - 3.f * a * a / 8.f;
    float f     = c + a * a * a / 8.f - 0.5f * a * b;
    float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;
    float xx[4] = { 1e10f, 1e10f, 1e10f, 1e10f };
    float delta;
    float h = 0.f;
    unsigned ireal = 0;

    // special case when f is zero
    if (fabs(f) < 1e-6f) 
    {
        delta = e * e - 4.f * g;
        if (delta < 0.f) return 0;
        delta = sqrt(delta);
        h  = 0.5f * (-e - delta);
        if (h >= 0.f)
        {
            h = sqrt(h);
            x[ireal++] = -h - 0.25f * a;
            x[ireal++] = h - 0.25f * a;
        }
        h = 0.5f * (-e + delta);
        if (h >= 0.f) 
        {
            h = sqrt(h);
            x[ireal++] = -h - 0.25f * a;
            x[ireal++] = h - 0.25f * a;
        }
        //Sort4(x);
        return ireal;
    }

    if (fabs(g) < 1e-6f) 
    {
        x[ireal++] = -0.25f * a;
        // this actually wants to solve a second order equation
        // we should specialize if it happens often
        unsigned ncubicroots = SolveCubic(0, e, f, xx, msk );
        // this loop is not nice
        for (unsigned i = 0; i < ncubicroots; i++) x[ireal++] = xx[i] - 0.25f * a;

        return ireal;
    }

    ireal = SolveCubic(2.f * e, e * e - 4.f * g, -f * f, xx, msk);

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
    float j = 0.5f * (e + h * h - f / h);
    ireal = 0;

    delta = h * h - 4.f * j;
    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        x[ireal++] = 0.5f * (-h - delta) - 0.25f * a;
        x[ireal++] = 0.5f * (-h + delta) - 0.25f * a;
    }
    delta = h * h - 4.f * g / j;
    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        x[ireal++] = 0.5f * (h - delta) - 0.25f * a;
        x[ireal++] = 0.5f * (h + delta) - 0.25f * a;
    }

    return ireal;
}


