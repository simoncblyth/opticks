


#ifdef __CUDACC__
__device__ __host__
#endif
static int SolveQuarticPureNeumark(Solve_t a, Solve_t b, Solve_t c, Solve_t d, Solve_t* rts, unsigned msk  )
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

    const Solve_t nought = 0.f ; 
    const Solve_t two = 2.f ; 
    const Solve_t four = 4.f ; 
    const Solve_t inv2 = 0.5f ; 

    const Solve_t asq = a*a ; 

    const Solve_t p = -two*b ; 
    const Solve_t q = b*b + a*c - four*d ; 
    const Solve_t r = (c-a*b)*c + asq*d  ;  

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
     



/*
    if (fabs(r) < 1e-3f)     
    {
        // Neumark p18 : constant term of (3.9) is zero -> resolvent cubic has one zero root
        // giving immediate factorization
        //    
        //     (A x**2 + B x + ( C - A*D/B )) * ( x**2 + D/B )  = 0 
        //     (  x**2 + a x + ( b - c/a ) ) * ( x**2 + c/a ) = 0     
        //
        //   

 
        int ireal = 0 ; 
 
        Solve_t coa = c/a ; // perfect sq term   
        if(coa >= 0.f )
        {
            Solve_t scoa = sqrt(coa) ; 
            rts[ireal++] = scoa ; 
            rts[ireal++] = -scoa ; 
        }
             
        Solve_t rdis = a * a - 4.f*(b-coa) ;
        if(rdis > 0.f)
        {
            Solve_t quad[2] ; 
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
*/

    Solve_t y = cubic_root_gem( p, q, r );

 
    if (y <= 0.f) return 0 ;

    const Solve_t bmy = b - y ;
    const Solve_t y4 = y*four ; 
    const Solve_t d4 = d*four ;
    const Solve_t bmysq = bmy*bmy ;
    const Solve_t gdis = asq - y4 ;
    const Solve_t hdis = bmysq - d4 ;
    if ( gdis < nought || hdis < nought ) return 0 ;

    // see Graphics Gems : Solving Quartics and Cubics for Graphics, Herbison-Evans

    const Solve_t g1 = a*inv2 ;
    const Solve_t h1 = bmy*inv2 ;
    const Solve_t gerr = asq + y4 ;        // p9: asq - 4y  ??
    const Solve_t herr = d > nought ? bmysq + d4 : hdis ; 

    Solve_t g2 ; 
    Solve_t h2 ; 

    if ( y < nought || herr*gdis > gerr*hdis )
    {
        const Solve_t gdisrt = sqrt(gdis) ;
        g2 = gdisrt*inv2 ;
        h2 = gdisrt != nought ? (a*h1 - c)/gdisrt : nought ;
    }
    else
    {
        const Solve_t hdisrt = sqrt(hdis) ;
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

    Solve_t h ;
    Solve_t hh ;
    Solve_t hmax ; 

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


    Solve_t g ; 
    Solve_t gg ; 
    Solve_t gmax ; 

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


    Solve_t v1[2],v2[2] ;

    Solve_t disc1 = gg*gg - four*hh ;
    Solve_t disc2 =  g*g - four*h ;

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


