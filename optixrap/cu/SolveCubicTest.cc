/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out

#include <cassert>
#include <complex> 
#include <cmath>


typedef double Solve_t ;
#include "fascending.h"

#include "Solve.h"
#include "SolveErrors.h"

#include <stdio.h>
#define rtPrintf printf


#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>


typedef std::complex<Solve_t> cplx ; 

struct CubicTest
{
    unsigned num  ; 
    unsigned nr  ; 
    unsigned msk ;
    bool coeff ; 
    bool comp ; 

    cplx  zr0[3] ;   // input roots 

    cplx  zco[4] ;   // complex coeffs

    Solve_t co[4] ;    // real coeff
    Solve_t dco[4] ;   // depressed cubic coeff

    Solve_t r1[3] ;   // result real roots
    Solve_t rterr[3] ; 
    Solve_t rtdel[3] ; 


    Solve_t delta ;   
    Solve_t disc ; 
    Solve_t sdisc ; 


    std::string desc_root(const Solve_t* a, unsigned n) const 
    {
         std::stringstream ss ; 
         for(unsigned i=0 ; i < n ; i++) ss << std::setw(10) << a[i] << " " ; 
         return ss.str();
    }
    std::string desc_root(const cplx* a, unsigned n) const 
    {
         std::stringstream ss ; 
         for(unsigned i=0 ; i < n ; i++) ss << std::setw(10) << a[i] << " " ; 
         return ss.str();
    }

    void dump()
    {

        int wi = 7 ; 

        std::cout 
            << " nr " << nr  ;
            ;

        std::cout << " zr0 " << desc_root(zr0, 3);
       
        std::cout 
            << " r1 " << desc_root(r1, nr )
            << " abc ( " << std::setw(wi) << co[2] << " " << std::setw(wi) << co[1] << " " << std::setw(wi) << co[0] << ") " 
            << " pq ( " << std::setw(wi) << dco[1] << " " << std::setw(wi) << dco[0] << ") " 
            << " delta " << delta
            << " disc " << disc
            << " sdisc " << sdisc
            << " " << SolveTypeMask(msk)
            << std::endl 
            ;

        dump_errors(); 

    }


    CubicTest(cplx z0, cplx z1, cplx z2, unsigned msk_, bool coeff_ ) : num(3), nr(0), msk(msk_), coeff(coeff_), comp(true)
    {
        assert(!coeff);
        zr0[0] = z0 ; 
        zr0[1] = z1 ; 
        zr0[2] = z2 ; 

        // see cubic.py 
        zco[0] = -z0*z1*z2 ;
        zco[1] =  z0*z1 + z0*z2 + z1*z2 ;
        zco[2] = -z0 - z1 - z2 ;
        zco[3] = cplx(1.f, 0.f) ; 

        co[0] = std::real(zco[0]);
        co[1] = std::real(zco[1]);
        co[2] = std::real(zco[2]);
        co[3] = std::real(zco[3]);

        assert( std::imag(zco[0]) == 0.f ) ;
        assert( std::imag(zco[1]) == 0.f ) ;
        assert( std::imag(zco[2]) == 0.f ) ;
        assert( std::imag(zco[3]) == 0.f ) ;

        solve(msk);    
    }

    CubicTest(Solve_t x0, Solve_t x1, Solve_t x2, unsigned msk_, bool coeff_ ) : num(3), nr(0), msk(msk_), coeff(coeff_), comp(false) 
    {
        if(coeff)
        {
            zr0[0] = 0 ; 
            zr0[1] = 0 ; 
            zr0[2] = 0 ; 
     
            co[0] = x0 ; 
            co[1] = x1 ; 
            co[2] = x2 ; 
            co[3] = 1.f ; 
        } 
        else
        {
            zr0[0] = x0 ; 
            zr0[1] = x1 ; 
            zr0[2] = x2 ; 
            //fascending_ptr( 3, r0 );

            co[0] = -x0*x1*x2 ;
            co[1] =  x0*x1 + x0*x2 + x1*x2 ;
            co[2] = -x0 - x1 - x2 ;
            co[3] = 1.f ;          
        }
        solve(msk);    
    }


    void dump_errors()
    {
        for(unsigned i=0 ; i < nr ; i++)
        {
            rtPrintf(" i %d "
                     "rt/err/del/frac (%9g %9g %9g ; %5g)\n"
                     ,
                     i 
                     ,
                     r1[i],rterr[i],rtdel[i], rterr[i]/r1[i]
                     );
        }
    }


    void solve(unsigned msk)
    { 
        //  Original cubic  :   x^3 + a*x^2 + b*x + c = 0     // 1,a,b,c
        //  Depressed cubic :   z^3 +   0   + p*z + q = 0     // 1,0,p,q   (from x->z-a/3 )   

        
        Solve_t a = co[2]; 
        Solve_t b = co[1]; 
        Solve_t c = co[0]; 


        nr = SolveCubicNumericalRecipe( a,b,c, r1, msk );
        //nr = SolveCubic(              a,b,c, r1, msk );

        //fascending_ptr( nr, r1 );

        cubic_errors( a, b, c, r1, rterr, rtdel, nr );


        // below for checking values of intermediates when precision goes to pot
        const Solve_t ott = 1.f / 3.f;
        const Solve_t sq3 = sqrt(3.f);
        const Solve_t inv6sq3    = 1.f / (6.f * sq3);

        const Solve_t p = b - a * a * ott;        
        const Solve_t q = c - a * b * ott + 2.f * a * a * a * ott * ott * ott;

        dco[0] = q ;
        dco[1] = p ;
        dco[2] = 0 ;
        dco[3] = 1 ;

        delta = 4.f * p * p * p + 27.f * q * q;
        disc = delta/(27.f*4.f) ; 
        sdisc = sqrt(disc) ;  


       //  Dividing delta by 27*4 yields the cubic discriminant:   
       //
       //        delta/(27*4 ) =  (p/3)**3 + (q/2)**2         
       //
       //  sqrt of discriminant is: 
       //
       //        sqrt(delta/(3*3*3*2*2)) = sqrt(delta)/(6*sqrt(3)) = sqrt(delta)*inv6sq3
       //

        dump();   
    }



};



void test_cubic( cplx r0, cplx r1, cplx r2, unsigned N, unsigned* msk, Solve_t* sc, unsigned smsk )
{ 
    std::cout << " r0 : " << r0 ; 
    std::cout << " r1 : " << r1 ; 
    std::cout << " r2 : " << r2 ; 
    std::cout << std::endl ; 

    for(unsigned i=0 ; i < N ; i++) CubicTest t(r0,r1,r2, msk[i], false) ;   
    std::cout << std::endl ; 

    if(sc) 
    {
        std::cout << " sc[0]: " << sc[0] ; 
        std::cout << " sc[1]: " << sc[1] ; 
        std::cout << " sc[2]: " << sc[2] ; 
        std::cout << std::endl ; 

        for(Solve_t s=sc[0] ; s < sc[1] ; s+=sc[2] )
        {
            for(unsigned i=0 ; i < N ; i++) CubicTest t(r0*( smsk & 1 ? s : 1) ,r1*(smsk & 2 ? s : 1),r2*(smsk & 4 ? s : 1), msk[i], false) ;   
            std::cout << std::endl ; 
        }
    }
}

void test_one_real_root(unsigned N, unsigned* msk, Solve_t* sc, unsigned smsk)
{
    std::cout << "test_one_real_root "  ;

    cplx r0(3,0) ;   // z**3 - 7.0*z**2 + 41.0*z - 87.0      one real root
    cplx r1(2,5) ;
    cplx r2(2,-5) ;

    test_cubic(r0,r1,r2, N, msk, sc, smsk);
}


void test_one_real_root_2(unsigned N, unsigned* msk, Solve_t* sc, unsigned smsk)
{
    std::cout << "test_one_real_root_2 "  ;

    cplx r0(300,0) ;   
    cplx r1(2,10) ;
    cplx r2(2,-10) ;

    test_cubic(r0,r1,r2, N, msk, sc, smsk);
}

void test_three_real_root(unsigned N, unsigned* msk, Solve_t* sc, unsigned smsk)
{
    std::cout << "test_three_real_root "  ;

    cplx r0(1,0) ; 
    cplx r1(2,0) ;
    cplx r2(3,0) ;

    test_cubic(r0,r1,r2, N, msk, sc, smsk );
}



void test_root_scan()
{
    static const unsigned N = 3 ; 
    unsigned msk[N] ; 
    msk[0] = SOLVE_VECGEOM ;
    msk[1] = SOLVE_UNOBFUSCATED  ;
    //msk[2] = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD_1 ;
    msk[2] = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ; 


//    test_one_real_root(N, msk, NULL, 0);
    test_one_real_root_2(N, msk, NULL, 0);
//    test_three_real_root(N, msk, NULL, 0);

    // control which roots to scale
    //unsigned smsk = 7 ; 
    //unsigned smsk = 6 ; 
    unsigned smsk = 0 ;  

    if(smsk > 0)
    {
        Solve_t sc[3] = {1., 1000., 100. };

        test_one_real_root(N, msk, sc, smsk);
        test_three_real_root(N, msk,sc, smsk);
    }
}



void test_coeff_artifact_ring_in_torus_hole_three_real_roots()
{
    unsigned msk  = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ; 

    Solve_t p,q,r ;   

    p = 49526.79994 ;        
    q = 408572956.1 ;
    r = -1483476.478 ;
    
    bool coeff = true ; 
    CubicTest( r, q, p,  msk, coeff ); 
}


int main()
{
    test_coeff_artifact_ring_in_torus_hole_three_real_roots();

    //test_root_scan();
 
    return 0 ; 
}


