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

// clang SolveQuarticTest.cc -lc++ && ./a.out && rm a.out

#include <cassert>
#include <complex> 
#include <cmath>
#include <iostream>

#include "fascending.h"

#define PURE_NEUMARK_DEBUG 1
#include "Solve.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>


typedef std::complex<float> cplx ; 

struct QuarticTest
{
    unsigned num  ; 
    unsigned nr  ; 
    unsigned msk ;
    bool coeff ; 
    bool comp ; 

    cplx  inroots[4] ;   // input roots 
    cplx  zco[5] ;   // complex coeffs
    float co[5] ;    // real coeff
    float roots[4] ;   // result real roots

    std::string desc_root(const float* a, unsigned n) const 
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
            << " inroots " << desc_root(inroots, 4)
            << " outroots " << nr  
            << " { " << desc_root(roots, nr ) << " } "
            << " abcd ( " 
            << std::setw(wi) << co[3] << " " 
            << std::setw(wi) << co[2] << " " 
            << std::setw(wi) << co[1] << " " 
            << std::setw(wi) << co[0] << ") " 
            << " " << SolveTypeMask(msk)
            << std::endl 
            ;
    }


    QuarticTest(cplx z0, cplx z1, cplx z2, cplx z3, unsigned msk_, bool coeff_ ) : num(4), nr(0), msk(msk_), coeff(coeff_), comp(true)
    {
        assert(!coeff);
        inroots[0] = z0 ; 
        inroots[1] = z1 ; 
        inroots[2] = z2 ; 
        inroots[3] = z3 ; 

        // see quartic.py 
        zco[0] =  z0*z1*z2*z3 ;
        zco[1] =  -z0*z1*z2 - z0*z1*z3 - z0*z2*z3 - z1*z2*z3 ;
        zco[2] = z0*z1 + z0*z2 + z0*z3 + z1*z2 + z1*z3 + z2*z3 ;
        zco[3] = -z0 - z1 - z2 - z3 ;
        zco[4] = cplx(1.f, 0.f) ; 

/*

from sympy import symbols, collect, expand   

In [6]: z0,z1,z2,z3,z = symbols("z0,z1,z2,z3,z")
In [7]: ezz = collect(expand((z-z0)*(z-z1)*(z-z2)*(z-z3)),z)
In [8]: ezz
Out[8]: z**4 + z**3*(-z0 - z1 - z2 - z3) + z**2*(z0*z1 + z0*z2 + z0*z3 + z1*z2 + z1*z3 + z2*z3) + z*(-z0*z1*z2 - z0*z1*z3 - z0*z2*z3 - z1*z2*z3) + z0*z1*z2*z3

*/


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

    void solve(unsigned msk)
    { 
        float a = co[3]; 
        float b = co[2]; 
        float c = co[1]; 
        float d = co[0]; 

        //nr = SolveQuartic( a,b,c,d, roots, msk );
        nr = SolveQuarticPureNeumark( a,b,c,d, roots, msk );
        fascending_ptr( nr, roots );

        dump();   
    }

};



void test_quartic( cplx z0, cplx z1, cplx z2, cplx z3,  unsigned N, unsigned* msk, float* sc, unsigned smsk )
{ 
    std::cout << " z0 : " << z0 ; 
    std::cout << " z1 : " << z1 ; 
    std::cout << " z2 : " << z2 ;
    std::cout << " z3 : " << z3 ;
    std::cout << std::endl ; 

    for(unsigned i=0 ; i < N ; i++) QuarticTest t(z0,z1,z2,z3, msk[i], false) ;   
    std::cout << std::endl ; 

    if(sc) 
    {
        std::cout << " sc[0]: " << sc[0] ; 
        std::cout << " sc[1]: " << sc[1] ; 
        std::cout << " sc[2]: " << sc[2] ; 
        std::cout << std::endl ; 

        for(float s=sc[0] ; s < sc[1] ; s+=sc[2] )
        {
            for(unsigned i=0 ; i < N ; i++) QuarticTest t(z0*( smsk & 1 ? s : 1) ,z1*(smsk & 2 ? s : 1),z2*(smsk & 4 ? s : 1), z3*(smsk & 8 ? s : 1), msk[i], false) ;   
            std::cout << std::endl ; 
        }
    }
}

void test_two_real_root(unsigned N, unsigned* msk, float* sc, unsigned smsk)
{
    std::cout << "test_two_real_root "  ;

    cplx z0(1,0) ;  
    cplx z1(2,0) ;
    cplx z2(3,0) ;
    cplx z3(4,0) ;

/*

In [5]: expand( (x-1)*(x-2)*(x-3)*(x-4) )
Out[5]: x**4 - 10*x**3 + 35*x**2 - 50*x + 24

*/


    test_quartic(z0,z1,z2,z3, N, msk, sc, smsk);
}

int main()
{
    static const unsigned N = 3 ; 
    unsigned msk[N] ; 

    unsigned best = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ; 

    msk[0] = SOLVE_VECGEOM ;
    msk[1] = SOLVE_UNOBFUSCATED  ;
    msk[2] = best  ;

    test_two_real_root(N, msk, NULL, 0);

    // control which roots to scale
    //unsigned smsk = 7 ; 
    //unsigned smsk = 6 ; 
    unsigned smsk = 0 ;  

    if(smsk > 0)
    {
        float sc[3] = {1., 1000., 100. };
        test_two_real_root(N, msk, sc, smsk);
    }

    return 0 ; 
}


