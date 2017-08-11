#include <optix.h>

typedef double Solve_t ; 
#include "SolveCubicNumericalRecipe.h"

/*
RT_CALLABLE_PROGRAM 
unsigned SolveCubicCallable(const Solve_t a, const Solve_t b, const Solve_t c, Solve_t* rts, unsigned msk)
{
    rts[0] = a*100. ; 
    rts[1] = b*100. ; 
    rts[2] = c*100. ; 
    return 3u  ; 
}
*/

RT_CALLABLE_PROGRAM 
unsigned SolveCubicCallable(const Solve_t a, const Solve_t b, const Solve_t c, Solve_t* rts, unsigned msk)
{
    return SolveCubic(a,b,c,rts,msk);
}


