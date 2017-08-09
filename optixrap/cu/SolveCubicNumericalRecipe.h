
#ifdef __CUDACC__
__device__ __host__
#endif
static unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned ) 
{
    //  p185 NUMERICAL RECIPES IN C 
    //  x**3 + a x**2 + b x + x = 0 

    const Solve_t zero(0) ; 
    const Solve_t one(1) ; 
    const Solve_t three(3) ; 
    const Solve_t othree = one/three ; 
    const Solve_t nine(9) ; 
    const Solve_t two(2) ; 
    const Solve_t twentyseven(27) ;
    const Solve_t fiftyfour(54) ;
    const Solve_t twpi = M_PI*two  ;
   
    const Solve_t a3 = a*othree ; 
    const Solve_t aa = a*a ; 
    const Solve_t Q = (aa - three*b)/nine ;                                         
    const Solve_t R = ((two*aa - nine*b)*a + twentyseven*c)/fiftyfour ;  // a,b,c real so Q,R real
    const Solve_t R2 = R*R ; 
    const Solve_t Q3 = Q*Q*Q ;
    const Solve_t R2_Q3 = R2 - Q3 ; 

    unsigned nr ; 

    if( R2_Q3 < zero ) // three real roots
    { 
         nr = 3 ; 
         const Solve_t theta = acos( R/sqrt(Q3) ); 
         const Solve_t qs = sqrt(Q); 

         xx[0] = -two*qs*cos(theta*othree) - a3 ;
         xx[1] = -two*qs*cos((theta+twpi)*othree) - a3 ;
         xx[2] = -two*qs*cos((theta-twpi)*othree) - a3 ; 
    }
    else
    {
         nr = 1 ; 
         const Solve_t A = -copysign(one, R)*cbrt( fabs(R) +  sqrt(R2_Q3) ) ; 
         const Solve_t B = A != zero ? Q/A : zero ; 

         xx[0] = (A + B) - a3  ;  
         xx[1] = zero ; 
         xx[2] = zero ; 
    }  


/*
#ifdef SOLVE_QUARTIC_DEBUG
    rtPrintf("// SOLVE_QUARTIC_DEBUG.SolveCubicNumericalRecipe "
             " abc (%g %g %g) " 
             " nr %u "
             "\n"
             ,
             a,b,c
             ,
             nr         
         );
#endif
*/


    return nr ; 
}


