#pragma once

#include "SolveEnum.h"

#include "SolveQuadratic.h"

// NB only one of the below SolveCubic must be uncommented
//#include "SolveCubicNumericalRecipe.h"
#include "SolveCubicPolyDivision.h"
//#include "SolveCubicDev.h"

/*
*SolveCubicNumericalRecipe*
    "cleaner" implementation, but get bizarre issue with OptiX (not pure CUDA) 
    seg-violation from cbrt(double) forcing use of cbrtf(float)
    BUT the lower precision in the cubic root of resolvent cubic causes 
    from the side torus artifacts ... so have to stick with the messy SolveCubic.h 
    for now

*/

#include "SolveQuartic.h"
//#include "SolveQuarticPureNeumark.h"


