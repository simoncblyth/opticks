

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



