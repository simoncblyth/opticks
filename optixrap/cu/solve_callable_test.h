__device__ void solve_callable_test()
{
    double a(10) ;
    double b(20) ;
    double c(30) ;
    double xx[3] ; 

    unsigned msk = 0u ; 
    unsigned nr = solve_callable[0](a,b,c,xx,msk);

    rtPrintf("solve_callable_test:solve_callable[0] abc (%g %g %g) nr %u xx (%g %g %g) \n",
             a,b,c,nr,xx[0],xx[1],xx[2]
            );
 

} 
