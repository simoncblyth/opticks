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

#pragma once

/*

Robust Quadratic Roots : Numerically Stable Method for Solving Quadratic Equations
======================================================================================



Reference
-------------

* https://people.csail.mit.edu/bkph/articles/Quadratics.pdf

Robustness in Geometric Computations
Christoph M. Hoffmann
Computer Science Purdue University

* http://www.cs.uoi.gr/~fudos/grad-material/robust4.pdf
* ~/opticks_refs/robust_geometry_calc.pdf 


* :google:`numerically robust line plane intersection`


Normal Quadratic
-----------------

   d t^2 + 2b t + c = 0  


  t =     -2b +- sqrt((2b)^2 - 4dc )
        -----------------------------
                2d

      -b +- sqrt( b^2 - d c )
      -----------------------
             d   


Alternative quadratic in 1/t 
--------------------------------


    c (1/t)^2 + 2b (1/t) + d  = 0 


    1/t  =   -2b +- sqrt( (2b)^2 - 4dc )
             ----------------------------
                      2c

    1/t  =    -b  +- sqrt( b^2 - d c )
             -------------------------
                      c

                     c
    t =    ---------------------------
             -b  +-  sqrt( b^2 - d c )


----------------

      q =  b + sign(b) sqrt( b^2 - d c )      

      q =  b + sqrt( b^2 - d c ) # b > 0
      q =  b - sqrt( b^2 - d c ) # b < 0
   

*/

static __device__
void robust_quadratic_roots(float& t1, float &t2, float& disc, float& sdisc, const float d, const float b, const float c)
{
    disc = b*b-d*c;
    sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // real roots for sdisc > 0.f 

#ifdef NAIVE_QUADRATIC
    t1 = (-b - sdisc)/d ;
    t2 = (-b + sdisc)/d ;  // t2 > t1 always, sdisc and d always +ve
#else
    // picking robust quadratic roots that avoid catastrophic subtraction 
    float q = b > 0.f ? -(b + sdisc) : -(b - sdisc) ; 
    float root1 = q/d  ; 
    float root2 = c/q  ;
    t1 = fminf( root1, root2 );
    t2 = fmaxf( root1, root2 );
#endif
} 


