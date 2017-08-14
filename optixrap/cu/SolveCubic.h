// NB only one of the below SolveCubic must be uncommented
//#include "SolveCubicNumericalRecipe.h"
//#include "SolveCubicPolyDivision.h"
//#include "SolveCubicDev.h"
#include "SolveCubicStrobachPolyFit.h"

/*

*SolveCubicPolyDivision*

*SolveCubicNumericalRecipe*
    "cleaner" implementation, but get bizarre issue with OptiX (not pure CUDA) 
    seg-violation from cbrt(double) forcing use of cbrtf(float)
    BUT the lower precision in the cubic root of resolvent cubic causes 
    from the side torus artifacts ... so have to stick with SolveCubicPolyDivision.h 
    for now

*SolveCubicStrobachPolyFit*
    Fast Iterative method from Peter Strobach, who in a paper claims
    that this approach of polynomial coeff fitting manages to handle
    very widely spaced roots which most root finders provide terribly 
    imprecise roots with


*/

