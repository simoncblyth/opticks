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

// /usr/local/env/geometry/quartic/quartic/strobach.cc




#ifdef __CUDACC__
__device__ __host__
#endif
void InitialGamma(const Solve_t& a, const Solve_t& b, const Solve_t& c, Solve_t& gamma )
{
    const Solve_t zero(0);
    const Solve_t one(1);
    const Solve_t two(2);
    const Solve_t otwo = one/two ;
    const Solve_t three(3);
    const Solve_t four(4);
    const Solve_t twentyseven(27);

    const Solve_t six(6);
    const Solve_t ott = one/three ; 

    const Solve_t sq3        = sqrt(three);
    const Solve_t inv6sq3    = one / (six * sq3);

    const Solve_t p = b - a * a * ott;                                       
    const Solve_t q = c - a * b * ott + two * a * a * a * ott * ott * ott;


    const Solve_t a3 = a/three ; 
    const Solve_t p3 = p/three ; 
    const Solve_t q2 = q/two ; 

    Solve_t delta = four * p * p * p +  twentyseven * q * q;   

    Solve_t t, u ;

    if (delta >= zero ) // only one real root,  Cardanos formula for the depressed cubic with -a/3 shift to yield original cubic root
    {
        delta = sqrt(delta);      
        Solve_t sdisc = delta*inv6sq3 ;

        t = q2 < zero ? -q2 + sdisc : q2 + sdisc ;  
            
        Solve_t tcu = copysign(one, t) * cbrt(fabs(t)) ; 
        Solve_t ucu = p3 / tcu ;        
               
        gamma  = q2 < zero ? -tcu + ucu + a3 : -ucu + tcu + a3  ;
    } 
    else 
    {
        delta = sqrt(-delta);
        t     = -otwo * q;
        u     = delta * inv6sq3;      // sqrt of negated discrim :  sqrt( -[(p/3)**3 + (q/2)**2] )

        gamma  = -two * sqrt(-p3) * cos(ott * atan2(u, t)) + a3  ; 
    }
} 



#ifdef __CUDACC__
__device__ __host__
#endif
unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned msk ) 
{
    Solve_t alfa,beta,gamma;
    InitialGamma(a,b,c,gamma);

    const Solve_t zero(0) ;
    const Solve_t two(2) ;

    Solve_t e1,e2,e3 ;
    Solve_t u1,u2 ;
    Solve_t q1,q2,q3 ;
    Solve_t d1,d2,d3 ;
    Solve_t ee,eee,eeee ;

    alfa = a - gamma ;
    beta = b - alfa*gamma ;

    e1 = zero ;
    e2 = zero ;
    e3 = c - gamma*beta ;

    eee=zero ;
    ee=zero ;

    for( int iter=0 ; iter < 16 ; iter++) 
    { 
        // --------

        u1 = alfa - gamma ;            // eqn (14)(15)
        u2 = beta - gamma*u1 ;

        q1 = e1 ;                      // eqn (17)(18)(19)
        q2 = e2 - gamma*q1 ;
        q3 = e3 - gamma*q2 ;
      
        d3 = u2 == zero ? zero : q3/u2 ;    // eqn (21)(22)(23)
        d2 = q2 - u1*d3 ;
        d1 = q1 - d3 ;

        // --------

        alfa += d1 ;     // eqn (7)(10)(12)
        beta += d2 ;
        gamma += d3 ;
 
        e1 = a - gamma - alfa ;       // eqn (5) fitting error [e1,e2,e3] ... difference of coeff
        e2 = b - alfa*gamma - beta ;
        e3 = c - gamma*beta ;

        eeee=eee ;              // ee before prior
        eee=ee ;                // prior ee
        ee=e1*e1+e2*e2+e3*e3 ;  // dot(e,e)    size of error^2

        if(ee == zero || ee == eee || ee == eeee) break ; 
    }
   
    // c--------------------- Solve Quadratic Equation ---------------------
    const Solve_t cc1 = alfa/two ;
    Solve_t diskr = cc1*cc1 - beta ;

    unsigned nr = diskr > zero ? 3u : 1u ; 

    xx[0] = -gamma  ;  
    if(nr == 3)
    { 
         diskr =sqrt(diskr) ;
         xx[1] = cc1 > zero ? -cc1-diskr  : -cc1+diskr  ; 
         xx[2] = beta/xx[0] ;
    }
    return nr ; 
}



