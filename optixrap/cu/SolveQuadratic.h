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



#ifdef __CUDACC__
__device__ __host__
#endif
static unsigned SolveQuadratic(Solve_t b, Solve_t c, Solve_t *rts, Solve_t disc, Solve_t offset )
{
/* 
     solve the quadratic equation :  x**2+b*x+c = 0 
        c=0 ->   x(x+b) = x**2 + b*x = 0  -> x=0, x=-b

*/
    Solve_t zero(0) ;
    Solve_t one(1) ;
    Solve_t two(2) ;
    Solve_t otwo = one/two ;
    unsigned nreal = 0 ;
    if(disc >= zero)
    {
        Solve_t sdisc = sqrt(disc) ;
        nreal = 2 ;
        rts[0] = b > zero ? -otwo*( b + sdisc) : -otwo*( b - sdisc)  ;
        rts[1] = rts[0] == zero ? -b : c/rts[0] ;
        rts[0] += offset ; 
        rts[1] += offset ; 
    }
    return nreal ;
}



