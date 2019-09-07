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


// nvcc -arch=sm_37 SolveCubicNumericalRecipeTest.cu -run ; rm a.out
// nvcc -arch=sm_30 SolveCubicNumericalRecipeTest.cu -run ; rm a.out
// nvcc -arch=sm_30 -I/Developer/OptiX/include SolveCubicNumericalRecipeTest.cu -run ; rm a.out

// https://stackoverflow.com/questions/8003166/usage-of-printf-in-cuda-4-0-compilation-error


// CUDART_INF_F
#include "math_constants.h"

//#include <optix_world.h>

#include <stdio.h>
#include <float.h>

typedef double Solve_t ; 
//#include "SolveCubicNumericalRecipe.h"
#include "SolveCubic.h"

__global__ void SolveCubicTest(double f)
{
    Solve_t p,q,r ;   

    p = 49526.79994 ;        
    q = 408572956.1 ;
    r = -1483476.478 ;

    Solve_t roq = -r/q ; 
 
    printf("SolveCubicTest pqr (%15g %15g %15g)  x^3 + p x^2 + q x + r = 0   -r/q %g   \n", p,q,r, roq );
  

    Solve_t xx[3] ; 
    unsigned nr = SolveCubic(p,q,r,xx, 0u ); 
    printf("nr %u  \n", nr ) ;
    for(unsigned i=0 ; i < nr ; i++)
    {
        Solve_t x = xx[i] ; 

        Solve_t x3 = x*x*x ; 
        Solve_t x2 = p*x*x ; 
        Solve_t x1 = q*x ; 
        Solve_t x0 = r ; 

        Solve_t x3_x2 = x3 + x2 ; 
        Solve_t x1_x0 = x1 + x0 ;
        Solve_t x3_x2_x1_x0 = x3_x2 + x1_x0 ;
  

        Solve_t residual = ((x + p)*x + q)*x + r ; 
        printf("xx[%u] = %15g  residual %15g  x3210 (%15g %15g %15g %15g) x3_x2 %15g x1_x0 %15g x3_x2_x1_x0 %15g    \n", i, xx[i], residual, x3, x2, x1, x0, x3_x2, x1_x0, x3_x2_x1_x0 ) ;
    }
}


int main()
{
  SolveCubicTest<<<1, 1>>>(1.2345);

  cudaDeviceReset();
  return 0;
}
