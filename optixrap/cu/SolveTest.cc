// clang SolveTest.cc -lc++ && ./a.out && rm a.out  
//
#include <math.h>
#include <stdio.h>
#define rtPrintf printf
#define SOLVE_QUARTIC_DEBUG 1
#define PURE_NEUMARK_DEBUG 1

typedef double Solve_t ;
#include "Solve.h"

void cubic_errors(Solve_t a,Solve_t b,Solve_t c, Solve_t* rts,Solve_t* rterr, Solve_t* rtdel,int nrts)
{
    Solve_t nought = 0.f ; 
    Solve_t doub4 = 4.f ; 
    Solve_t doub3 = 3.f ; 
    Solve_t doub2 = 2.f ; 
    Solve_t doub12 = 12.f ; 
    Solve_t doub6 = 6.f ; 
    Solve_t doub24 = 24.f ; 
/*

In [24]: ex = x**3 + a*x**2 + b*x + c

In [25]: diff(ex,x)
Out[25]: 2*a*x + b + 3*x**2

In [26]: diff(diff(ex,x),x)
Out[26]: 2*a + 6*x

In [27]: diff(diff(diff(ex,x),x),x)
Out[27]: 6
    
*/ 
    for ( int k = 0 ; k < nrts ; ++ k ) 
    {   
        rtdel[k] = ((rts[k]+a)*rts[k]+b)*rts[k]+c ;

        if (rtdel[k] == nought) 
        { 
            rterr[k] = nought;
        }
        else
        {   
            Solve_t deriv = (doub3*rts[k]+doub2*a)*rts[k]+b  ;
            if (deriv != nought)
            {
                rterr[k] = fabs(rtdel[k]/deriv);
            }
            else
            {   
               deriv = doub6*rts[k]+doub2*a  ;
               if (deriv != nought)
               {
                   rterr[k] = sqrt(fabs(rtdel[k]/deriv)) ;
               }
            }   
         }   
      }   
}


void quartic_errors(Solve_t a,Solve_t b,Solve_t c,Solve_t d, Solve_t* rts,Solve_t* rterr, Solve_t* rtdel,int nrts)
{
    Solve_t nought = 0.f ; 
    Solve_t doub4 = 4.f ; 
    Solve_t doub3 = 3.f ; 
    Solve_t doub2 = 2.f ; 
    Solve_t doub12 = 12.f ; 
    Solve_t doub6 = 6.f ; 
    Solve_t doub24 = 24.f ; 
 
    /*

In [18]: ex
Out[18]: a*x**3 + b*x**2 + c*x + d + x**4

In [19]: diff(ex,x)
Out[19]: 3*a*x**2 + 2*b*x + c + 4*x**3

In [20]: diff(diff(ex,x),x)
Out[20]: 6*a*x + 2*b + 12*x**2

In [21]: diff(diff(diff(ex,x),x),x)
Out[21]: 6*a + 24*x

In [22]: diff(diff(diff(diff(ex,x),x),x),x)
Out[22]: 24

     
    */

    for ( int k = 0 ; k < nrts ; ++ k ) 
    {   
        rtdel[k] = (((rts[k]+a)*rts[k]+b)*rts[k]+c)*rts[k]+d ;

        if (rtdel[k] == nought) 
        { 
            rterr[k] = nought;
        }
        else
        {   
            Solve_t deriv = ((doub4*rts[k]+doub3*a)*rts[k]+doub2*b)*rts[k]+c ;
            if (deriv != nought)
            {
                rterr[k] = fabs(rtdel[k]/deriv);
            }
            else
            {   
               deriv = (doub12*rts[k]+doub6*a)*rts[k]+doub2*b ;
               if (deriv != nought)
               {
                   rterr[k] = sqrt(fabs(rtdel[k]/deriv)) ;
               }
               else
               {   
                   deriv = doub24*rts[k]+doub6*a ;
                   rterr[k] = deriv != nought ?   cbrt(fabs(rtdel[k]/deriv)) : sqrt(sqrt(fabs(rtdel[k])/doub24)) ; 
               }   
            }   
         }   
      }   

} 



void test_cubic(Solve_t p, Solve_t q, Solve_t r)
{
    unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ;

    Solve_t xx[3] ; 
    unsigned ireal = SolveCubic(p, q, r, xx, msk);
 
    //Solve_t crr  = cubic_real_root( p,q,r, msk );  // tis giving sqrt of largest root ?
    //Solve_t clrr = cubic_lowest_real_root( p,q,r ) ;  // gives largest root
    //Solve_t crr2 = crr*crr ;

    rtPrintf("test_cubic"
             " pqr (%g,%g,%g) "
             " xx (%g,%g,%g)  "
            "\n"
             ,
             p, q, r
             ,
             xx[0],xx[1],xx[2]

            );

    Solve_t rterr[3] ; 
    Solve_t rtdel[3] ; 
    cubic_errors( p, q, r, xx, rterr, rtdel, ireal );

    for(unsigned i=0 ; i < ireal ; i++)
    {
        rtPrintf(" i %d "
                 "rt/err/del/frac (%9g %9g %9g ; %5g)\n"
                 ,
                 i 
                 ,
                 xx[i],rterr[i],rtdel[i], rterr[i]/xx[i]
                 );
    }
}


void dump_quartic( Solve_t a, Solve_t b, Solve_t c, Solve_t d, Solve_t* xx, int nxx)
{
    Solve_t rterr[4] ; 
    Solve_t rtdel[4] ; 

    quartic_errors( a, b, c, d, xx, rterr, rtdel, nxx );

    rtPrintf("dump_quartic"
             " abcd (%g,%g,%g,%g) "
             " nxx %d "
             " xx (%g,%g,%g,%g)  "
            "\n"
             ,
             a, b, c, d
             ,
             nxx
             ,
             xx[0],xx[1],xx[2],xx[3]
            );

    for(unsigned i=0 ; i < nxx ; i++)
    {
        rtPrintf(" i %d "
                 "rt/err/del/frac (%9g %9g %9g ; %5g)\n"
                 ,
                 i 
                 ,
                 xx[i],rterr[i],rtdel[i], rterr[i]/xx[i]
                 );
    }
}


void test_quartic(Solve_t a, Solve_t b, Solve_t c, Solve_t d)
{
    unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ;

    Solve_t xx[4] ; 
    unsigned nxx = SolveQuartic(a, b, c, d, xx, msk);
    dump_quartic(a,b,c,d, xx, nxx);

    Solve_t yy[4] ; 
    unsigned nyy = SolveQuarticPureNeumark(a, b, c, d, yy, msk);
    dump_quartic(a,b,c,d, yy, nyy);
}


int main()
{
    //test_cubic( -6.f, 11.f, -6.f );  // expand( (x-1)*(x-2)*(x-3) )  ->   x**3 - 6*x**2 + 11*x - 6
    //test_cubic( 33.0401f,90.0058f,-0.000324894f) ;
    //test_quartic( -10.f, 35.f, -50.f, 24.f ); // expand( (x-1)*(x-2)*(x-3)*(x-4) )  -> x**4 - 10*x**3 + 35*x**2 - 50*x + 24

    test_quartic( 0.f, -22100.f, 0.f, 121000000.f ); // expand( (x+100)*(x+110)*(x-100)*(x-110) )   x**4 - 22100*x**2 + 121000000

/*  
   
    //  A*x^4 + B*x^3 + C*x^2 + D*x + E = 0 
    //    x^4 + a*x^3 + b*x^2 + c*x + d = 0 
    //
    //    B*C*D - B*B*E - A*D*D = 0 
    //    a*b*c - a*a*d - 1*c*c = 0       
    //
    //     c^2 + -a*b c^0 + a*a*d = 0 
    //
    //      c = (a*b + sqrt( a*b*a*b - 4*a*a*d ))/2.0f
    //
    //       
    //                             set a,b,d -> 1   
    //                             c - 1 - c^2 = 0 -> c^2 - c + 1 = 0    c = 1 +- sqrt(-ve)
    //    
    //                             set a,b,d -> 1,4,-1
    //                              4c + 1 -c^2 = 0     c^2 - 4c - 1 = 0   c =  4 +- sqrt(16 + 1) / 2
    //  

    Solve_t a = 1.f ; 
    Solve_t b = 4.f ; 
    Solve_t d = -1.f ; 
    Solve_t c = c = (a*b + sqrt( a*b*a*b - 4.f*a*a*d ))/2.f ; 
    
     
    test_quartic( a, b, c, d );    
*/

    return 0 ; 
}


/*
test_cubic : Big residuals all? having small negative resolvent cubic r    

 ireal 4 i 3 root -3.88475 residual 2799.44  dis12 ( 4.07639 179.478 ) h 0.000971258  pqr (33.0401 90.0058 -0.000324894 )  j g/j (-1.0191 -44.8695 )  
 ireal 4 i 3 root -1.60037 residual 746.658  dis12 ( 9.4239 77.8749 ) h 0.00112281  pqr (33.093 90.3149 -0.00056974 )  j g/j (-2.35597 -19.4687 )  
 ireal 4 i 2 root 8.6907 residual 1804.84  dis12 ( 5.27271 138.008 ) h 0.000791402  pqr (32.9529 89.5536 -0.000228793 )  j g/j (-1.31818 -34.5021 )  
 ireal 4 i 3 root -3.05701 residual 1803.86  dis12 ( 5.27271 138.008 ) h 0.000791402  pqr (32.9529 89.5536 -0.000228793 )  j g/j (-1.31818 -34.5021 )  
 ireal 4 i 0 root -0.709157 residual 396.827  dis12 ( 45.2559 20.6214 ) h 0.000515888  pqr (37.2095 112.826 -0.000452474 )  j g/j (-11.314 -5.15535 )  
 ireal 4 i 1 root 6.01809 residual 396.827  dis12 ( 45.2559 20.6214 ) h 0.000515888  pqr (37.2095 112.826 -0.000452474 )  j g/j (-11.314 -5.15535 )  
 ireal 4 i 2 root 4.92552 residual 180.901  dis12 ( 45.2559 20.6214 ) h 0.000515888  pqr (37.2095 112.826 -0.000452474 )  j g/j (-11.314 -5.15535 )  
 ireal 4 i 2 root 8.61213 residual 1974.74  dis12 ( 6.55906 141.824 ) h 0.00124027  pqr (37.1539 112.545 -0.000734833 )  j g/j (-1.63976 -35.4559 )  
 ireal 4 i 3 root 0.384444 residual 180.737  dis12 ( 45.2559 20.6214 ) h 0.000515888  pqr (37.2095 112.826 -0.000452474 )  j g/j (-11.314 -5.15535 )  
 ireal 4 i 3 root -3.29684 residual 1973.1  dis12 ( 6.55906 141.824 ) h 0.00124027  pqr (37.1539 112.545 -0.000734833 )  j g/j (-1.63976 -35.4559 )  
 ireal 4 i 2 root 16.1244 residual 34295  dis12 ( 1.02297 708.044 ) h 0.00125725  pqr (32.8843 89.2673 -0.000454326 )  j g/j (-0.255742 -177.011 )  
 ireal 4 i 3 root -10.4847 residual 34282.1  dis12 ( 1.02297 708.044 ) h 0.00125725  pqr (32.8843 89.2673 -0.000454326 )  j g/j (-0.255742 -177.011 )  
 ireal 4 i 2 root 8.57149 residual 1927.92  dis12 ( 6.6322 139.783 ) h 0.001105  pqr (37.0887 112.126 -0.000583498 )  j g/j (-1.65805 -34.9458 )  
 ireal 4 i 3 root -3.2515 residual 1926.48  dis12 ( 6.6322 139.783 ) h 0.001105  pqr (37.0887 112.126 -0.000583498 )  j g/j (-1.65805 -34.9458 )  
 ireal 4 i 0 root 0.597041 residual 150.637  dis12 ( 19.8605 36.076 ) h 0.000557529  pqr (32.71 88.3634 -0.000214762 )  j g/j (-4.96512 -9.01901 )  
 ireal 4 i 1 root 5.05355 residual 150.637  dis12 ( 19.8605 36.076 ) h 0.000557529  pqr (32.71 88.3634 -0.000214762 )  j g/j (-4.96512 -9.01901 )  
 ireal 4 i 2 root 5.82902 residual 273.73  dis12 ( 19.8605 36.076 ) h 0.000557529  pqr (32.71 88.3634 -0.000214762 )  j g/j (-4.96512 -9.01901 )  
 ireal 4 i 3 root -0.177312 residual 273.527  dis12 ( 19.8605 36.076 ) h 0.000557529  pqr (32.71 88.3634 -0.000214762 )  j g/j (-4.96512 -9.01901 )  
 ireal 4 i 2 root 13.606 residual 16614  dis12 ( 1.92631 478.93 ) h 0.00167669  pqr (37.0051 111.703 -0.00106524 )  j g/j (-0.481577 -119.733 )  
 ireal 4 i 2 root 8.92594 residual 2319.92  dis12 ( 5.87294 156.809 ) h 0.00124101  pqr (36.9679 111.424 -0.000706655 )  j g/j (-1.46824 -39.2024 )  
 ireal 4 i 3 root -8.2785 residual 16603.8  dis12 ( 1.92631 478.93 ) h 0.00167669  pqr (37.0051 111.703 -0.00106524 )  j g/j (-0.481577 -119.733 )  
 ireal 4 i 3 root -3.59642 residual 2318.08  dis12 ( 5.87294 156.809 ) h 0.00124101  pqr (36.9679 111.424 -0.000706655 )  j g/j (-1.46824 -39.2024 )  

*/




