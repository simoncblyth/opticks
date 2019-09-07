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

// NB only one of the below SolveCubic must be uncommented

//#include "SolveCubicPolyDivision.h"
#include "SolveCubicStrobachPolyFit.h"
//#include "SolveCubicNumericalRecipe.h"
//#include "SolveCubicDev.h"

/*
*SolveCubicPolyDivision*
    current default

    Slightly different approach compared to SolveCubicNumericalRecipe 
    with a bit less double precision trig (sqrt,copysign,fabs,cbrt,cos,atan2) 
    that appears to avoid the issues with SolveCubicNumericalRecipe.

*SolveCubicStrobachPolyFit*
    Fast Iterative method from Peter Strobach, who in a paper claims
    that this approach of polynomial coeff fitting manages to handle
    very widely spaced roots which most root finders provide terribly 
    imprecise roots with

*SolveCubicNumericalRecipe*
    "cleaner" implementation, but lots of heavy trig

    Get bizarre issue with OptiX (not pure CUDA) 
    seg-violation from cbrt(double) forcing use of cbrtf(float)
    BUT the lower precision in the cubic root of resolvent cubic causes 
    from the side torus artifacts ... so have to stick with SolveCubicPolyDivision.h 
    for now

    Appears that lots of double precision trig/cbrt is very heavy on GPU
    Is the problem coming from acos ? 

    Double precision trig heavy closed form
    (acos,sqrt,cos,fabs,cbrt,copysign) causing bizarreness::

    * also giving rtContextCompile errors, segv sometimes, sometimes Assertion failed: "!name.empty()"


*SolveCubicDev*
    early version of SolveCubicPolyDivision with lots of mask switches, 
    used for initial dev (not recommended)


*/

