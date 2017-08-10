//#pragma once

#ifdef __CUDACC__
__device__ __host__
#endif
unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned msk ) 
{
    //  p185 NUMERICAL RECIPES IN C 
    //  x**3 + a x**2 + b x + x = 0 

    const Solve_t zero(0) ; 
    const Solve_t one(1) ; 
    const Solve_t two(2) ; 
    const Solve_t three(3) ; 
    const Solve_t nine(9) ; 
    const Solve_t twentyseven(27) ;
    const Solve_t fiftyfour(54) ;

    const Solve_t othree = one/three ; 
    const Solve_t otwo = one/two ; 
    const Solve_t twpi = M_PI*two  ;
   
    const Solve_t a3 = a*othree ; 
    const Solve_t aa = a*a ; 

    const Solve_t Q = (aa - three*b)/nine ;                                         
    const Solve_t R = ((two*aa - nine*b)*a + twentyseven*c)/fiftyfour ;  // a,b,c real so Q,R real
    const Solve_t R2 = R*R ; 
    const Solve_t Q3 = Q*Q*Q ;
    const Solve_t R2_Q3 = R2 - Q3 ; 
  
    unsigned nr = R2_Q3 < zero ? 3 : 1 ; 

    if( nr == 3 ) // three real roots
    { 
         const Solve_t theta = acos( R/sqrt(Q3) ); 
         const Solve_t qs = sqrt(Q); 

         xx[0] = -two*qs*cos(theta*othree) - a3 ;
         xx[1] = -two*qs*cos((theta+twpi)*othree) - a3 ;
         xx[2] = -two*qs*cos((theta-twpi)*othree) - a3 ; 
    }
    else
    { 
         const Solve_t R_R2_Q3 = fabs(R) + sqrt(R2_Q3) ; 

         // OptiX prone to segv in createPTXFromFile with either cbrt or pow of double, works OK pure CUDA
         const Solve_t croot = cbrt( R_R2_Q3 ) ; 

         //const Solve_t croot = cbrtf( R_R2_Q3 ) ;   // float works, but means conversions, lower prec, torus artifacts
         //const Solve_t croot = pow( R_R2_Q3, othree ) ; 
         //const Solve_t croot = one/rcbrt( R_R2_Q3 ) ;  

         const Solve_t A = -copysign(one, R)*croot  ; 
         const Solve_t B = A == zero ? zero : Q/A ; 

         xx[0] = A + B - a3  ;  
         xx[1] = zero ; 
         xx[2] = zero ; 
    }  

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
    return nr ; 
}


