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

//#pragma once

#ifdef __CUDACC__
__device__ __host__
#endif
unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned msk ) 
{
    //  p185 NUMERICAL RECIPES IN C 
    //  x**3 + a x**2 + b x + x = 0 

    const Solve_t zero(0) ; 
    const Solve_t one(1) ; 
    const Solve_t two(2) ; 
    const Solve_t three(3) ; 
    const Solve_t nine(9) ; 
    const Solve_t twentyseven(27) ;
    const Solve_t fiftyfour(54) ;

    const Solve_t othree = one/three ; 
    //const Solve_t otwo = one/two ; 
    const Solve_t twpi = M_PI*two  ;
   
    const Solve_t a3 = a*othree ; 
    const Solve_t aa = a*a ; 

    const Solve_t Q = (aa - three*b)/nine ;                                         
    const Solve_t R = ((two*aa - nine*b)*a + twentyseven*c)/fiftyfour ;  // a,b,c real so Q,R real
    const Solve_t R2 = R*R ; 
    const Solve_t Q3 = Q*Q*Q ;
    const Solve_t R2_Q3 = R2 - Q3 ; 
  
    unsigned nr = R2_Q3 < zero ? 3 : 1 ; 

    // GPU: really dislikes smth about the below...
    // OptiX prone to segv in createPTXFromFile with either cbrt or pow of double, works OK pure CUDA

    if( nr == 3 ) // three real roots
    { 

         const Solve_t theta = acos( R/sqrt(Q3) ); 
         //const Solve_t theta = atan2( sqrt(-R2_Q3), -R );   // UNVALIDATED ATTEMPT TO REPLACE acos with atan2 (see below)
         const Solve_t qs = sqrt(Q); 

         xx[0] = -two*qs*cos(theta*othree) - a3 ;
         xx[1] = -two*qs*cos((theta+twpi)*othree) - a3 ;
         xx[2] = -two*qs*cos((theta-twpi)*othree) - a3 ; 
    }
    else  // one real root
    { 
         const Solve_t R_R2_Q3 = fabs(R) + sqrt(R2_Q3) ; 
         const Solve_t croot = cbrt( R_R2_Q3 ) ; 
         const Solve_t A = -copysign(one, R)*croot  ; 
         const Solve_t B = A == zero ? zero : Q/A ; 

         xx[0] = A + B - a3  ;  
         xx[1] = zero ; 
         xx[2] = zero ; 
    }  

#ifdef SOLVE_QUARTIC_DEBUG
    rtPrintf("// SOLVE_QUARTIC_DEBUG.SolveCubicNumericalRecipe "
             " abc (%g %g %g) " 
             " nr %u "
             "\n"
             ,
             a,b,c
             ,
             nr         
         );
#endif
    return nr ; 
}





/*

Suspect acos implicated in OptiX wierdness/crashes : replace with atan2 ?
=============================================================================

Try to see correspondence between PolyDivision p,q,delta and NumericalRecipe Q,R approaches:


         3*b - a^2 
    p =  ----------      ( -p/3 = -p3 = Q )
             3

    q =  27*c -  9*a*b + 2*a^3      ( q/2 = R )
       ------------------------
                27

    delta = 4*p^3 + 27*q^2

          = -4*27*Q^3 + 27*4*R^2

         =  27*4* ( R^2 - Q^3 )       // R^2 < Q^3 -> delta -ve


   sqrt(-delta) = 2*3*sqrt(3)*sqrt( Q^3 - R^2 )



Three real root
-------------------

  t =  -q/2    =  -R

       sqrt(-delta)   
  u =  ------------     =  sqrt( Q^3 - R^2 )  
       2*3*sqrt(3)


   atan2( u, t )        = atan2( sqrt( Q^3 - R^2 ), -R )


   u^2 + t^2   =  Q^3 - R^2 + R^2   =  Q^3 


                  *
                 /|
  sqrt(Q^3)     / | 
               /  | 
              /   | t  (-R)
             /    |
            /     |
           *------*
              u

    sqrt(Q^3 - R^2)


   acos( R/sqrt(Q^3) ) 



        x[0]  = two * sqrt(-p3) * cos(ott * atan2(u, t)) - a3  ; 

         //const Solve_t theta = acos( R/sqrt(Q3) ); 
         const Solve_t theta = atan2( sqrt(-R2_Q3), -R );   // see below
         const Solve_t qs = sqrt(Q); 

         xx[0] = -two*qs*cos(theta*othree) - a3 ;


   sign diff in cosine
 


* https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/

acos(-x) = PI – acos(x)
asin(-x) = -asin(x)
asin(x) + acos(x) = PI/2
atan(-x) = -atan(x)

 
*/




