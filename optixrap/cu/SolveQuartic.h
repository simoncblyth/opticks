/*
Adapting polynomial solvers from ./volumes/kernel/TorusImplementation2.h
attempting to get operational inside OptiX 
*/



#ifdef __CUDACC__
__device__ __host__
#endif
static Solve_t cubic_sqroot(Solve_t p, Solve_t q, Solve_t r, unsigned msk)
{
    const Solve_t zero(0); 
    Solve_t xx[3] ; 
    unsigned num_real_roots = SolveCubic(p, q, r, xx, msk);
    Solve_t h = num_real_roots == 1 ?  xx[0] : ( xx[0] >= zero ? xx[0] : ( xx[1] >= zero ? xx[1] : xx[2] )) ; 
    return h <= zero ? zero : sqrt(h) ; 
}



#ifdef __CUDACC__
__device__ __host__
#endif
static int SolveQuartic(Solve_t a, Solve_t b, Solve_t c, Solve_t d, Solve_t *x, unsigned msk  )
{
    // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    // Input: a,b,c,d
    // Output: x[4] - real solutions
    // Returns number of real solutions (0 to 3)


    const Solve_t zero(0) ; 
    const Solve_t one(1) ; 
    const Solve_t two(2) ; 
    const Solve_t otwo(one/two) ; 
    const Solve_t three(3) ; 
    const Solve_t four(4) ; 
    const Solve_t eight(8) ; 
    const Solve_t sixteen(16) ; 
    const Solve_t twofivesix(256) ; 

    const Solve_t aa = a*a ; 
    const Solve_t a4 = a/four ; 

    // (1,0,e,f,g) are coeff of depressed quartic
    const Solve_t e     = b - three * aa / eight ;
    const Solve_t f     = c + a * aa / eight - otwo * a * b;
    const Solve_t g     = d - three * aa * aa / twofivesix + aa * b / sixteen - a * c / four ;

    //  (1,p,q,r) are coeffs of resolvent cubic
    const Solve_t p = two * e ;
    const Solve_t q = e * e - four * g ;
    const Solve_t r = -f * f ;

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

    Solve_t xx[4] ;
    Solve_t delta;
    Solve_t h = zero ;
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
#ifdef SOLVE_QUARTIC_DEBUG
       rtPrintf("// SOLVE_QUARTIC_DEBUG small-r %g \n", r );
#endif

        delta = e * e - four * g;
 
        Solve_t quad[2] ; 
        int iquad = SolveQuadratic( e,  g , quad, delta, zero ) ; 
        for(int i=0 ; i < iquad ; i++)
        {
            h = quad[i] ;
            if(h >= zero)
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
#ifdef SOLVE_QUARTIC_DEBUG
       rtPrintf("// SOLVE_QUARTIC_DEBUG small-g %g \n", g );
#endif

        x[ireal++] = -a4 ;   // z=0 root

        unsigned ncubicroots = SolveCubic(zero, e, f, xx, msk );   // 0 as z**2 term already zero

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

    //h = cubic_root_gem( p, q, r );
    h = cubic_sqroot( p, q, r, msk );  // sqrt of cubic root

#ifdef SOLVE_QUARTIC_DEBUG
    rtPrintf("// SOLVE_QUARTIC_DEBUG cubic_sqroot h %g \n", h );
#endif

     if (h <= zero) return 0;

  
    Solve_t j = otwo * (e + h * h - f / h);   

    ireal = 0;

    Solve_t dis1 = h * h - four * j;  // discrim of 1st factored quadratic

    ireal += SolveQuadratic( h, j , x, dis1, -a4 ) ; 

    Solve_t dis2 = h * h - four * g / j;    // discrim of 2nd factored quadratic

    ireal += SolveQuadratic( -h, g/j , x+ireal, dis2, -a4 ) ; 


#ifdef SOLVE_QUARTIC_DEBUG
    for(int i=0 ; i < ireal ; i++)
    {
        Solve_t residual = (((x[i] + a)*x[i] + b)*x[i] + c)*x[i] + d ; 
        if(residual > 100.f )
        {
           rtPrintf(
                 "// SOLVE_QUARTIC_DEBUG solve-exit "
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
#endif

    return ireal;
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


/*
                 " efg (%g %g %g ) "  
                 e,f,g


                 " abcd (%g %g %g %g ) "  
                 a,b,c,d
                 ,


*/

