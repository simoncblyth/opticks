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

#include <optix.h>

typedef double Solve_t ; 
#include "SolveCubicNumericalRecipe.h"

RT_CALLABLE_PROGRAM 
unsigned cbrtTestCallable(const Solve_t a, const Solve_t b, const Solve_t c, Solve_t* rts, unsigned msk)
{
/*
    rts[0] = a*100. ; 
    rts[1] = b*100. ; 
    rts[2] = c*100. ; 

    return 3  ; 
*/
    return SolveCubic(a,b,c,rts,msk);
}


