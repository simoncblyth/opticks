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

