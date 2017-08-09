


#ifdef __CUDACC__
__device__ __host__
#endif
static unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t *x, unsigned msk ) 
{
    // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
    //                                                   a2      a1    a0
    // Input: a,b,c
    // Output: x[3] real solutions
    // Returns number of real solutions (1 or 3)


    const Solve_t ott        = 1.f / 3.f; 
    const Solve_t sq3        = sqrt(3.f);
    const Solve_t inv6sq3    = 1.f / (6.f * sq3);
    unsigned int ireal = 1;

    const Solve_t p = b - a * a * ott;                                       
    const Solve_t q = c - a * b * ott + 2.f * a * a * a * ott * ott * ott;
    const Solve_t p3 = p/3.f ; 
    //const Solve_t p33 = p3*p3*p3 ;  
    const Solve_t q2 = q/2.f ; 

    //const Solve_t q22 = q2*q2 ; 
    //const Solve_t disc = p33 + q22 ;  


    Solve_t delta = 4.f * p * p * p + 27.f * q * q;   

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

    Solve_t t, u ;

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
            Solve_t sdisc = delta*inv6sq3 ;

            if( msk & SOLVE_ROBUST_VIETA )
            {
                t = q2 < 0.f ? -q2 + sdisc : q2 + sdisc ;  
            
                Solve_t tcu = copysign(1.f, t) * cbrt(fabs(t)) ; 
                Solve_t ucu = p3 / tcu ;        
               
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

                Solve_t tcu = copysign(1.f, t) * cbrt(fabs(t)) ;
                Solve_t ucu = copysign(1.f, u) * cbrt(fabs(u)) ; 
      
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
            Solve_t tmp = u > 0.f ? 0.5f*(-u - delta) : 0.5f*(-u + delta) ; 
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







