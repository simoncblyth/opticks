// /usr/local/env/geometry/quartic/quartic/strobach.cc

#ifdef __CUDACC__
__device__ __host__
#endif
void ApproxInitialGammaF(const Solve_t& a, const Solve_t& b, const Solve_t& c, Solve_t& gamma )
{
    // NB using float version of of expensive stuff like cbrtf, 
    //    as just need initial estimate of gamma which subsequently gets refined by the polyfit 
    //

    const Solve_t zero(0) ; 
    const Solve_t one(1) ; 
    const Solve_t two(2) ; 
    const Solve_t three(3) ; 
    const Solve_t nine(9) ; 
    const Solve_t twentyseven(27) ;
    const Solve_t fiftyfour(54) ;
    const Solve_t othree = one/three ; 

    const Solve_t a3 = a*othree ; 
    const Solve_t aa = a*a ; 

    const Solve_t Q = (aa - three*b)/nine ;                                         
    const Solve_t R = (a*(two*aa - nine*b) + twentyseven*c)/fiftyfour ;  // a,b,c real so Q,R real
    const Solve_t R2 = R*R ; 
    const Solve_t Q3 = Q*Q*Q ;
    const Solve_t R2_Q3 = R2 - Q3 ; 

    if( R2_Q3 < zero ) // three real roots
    { 
         const Solve_t theta = acos( R/sqrt(Q3) ); 
         const Solve_t qs = sqrt(Q); 
         gamma = two*qs*cos(theta*othree) - a3 ;
    }
    else
    { 
         const Solve_t R_R2_Q3 = fabs(R) + sqrt(R2_Q3) ; 
         const Solve_t croot = cbrt( R_R2_Q3 ) ; 
         const Solve_t A = -copysign(one, R)*croot  ; 
         const Solve_t B = A == zero ? zero : Q/A ; 
         gamma = -A - B + a3  ;
   }  
} 



#ifdef __CUDACC__
__device__ __host__
#endif
unsigned SolveCubic(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned msk ) 
{
    Solve_t alfa,beta,gamma;
    ApproxInitialGammaF(a,b,c,gamma);

    Solve_t zero(0) ;
    Solve_t e1,e2,e3 ;
    Solve_t u1,u2 ;
    Solve_t q1,q2,q3 ;
    Solve_t d1,d2,d3 ;
    Solve_t ee,eee,eeee ;

    alfa = a - gamma ;
    beta = b - alfa*gamma ;

    e1 = zero ;
    e2 = zero ;
    e3 = c - gamma*beta ;

    eee=zero ;
    ee=zero ;

    for( int iter=0 ; iter < 16 ; iter++) 
    { 
        // --------

        u1 = alfa - gamma ;            // eqn (14)(15)
        u2 = beta - gamma*u1 ;

        q1 = e1 ;                      // eqn (17)(18)(19)
        q2 = e2 - gamma*q1 ;
        q3 = e3 - gamma*q2 ;
      
        d3 = u2 == zero ? zero : q3/u2 ;    // eqn (21)(22)(23)
        d2 = q2 - u1*d3 ;
        d1 = q1 - d3 ;

        // --------

        alfa += d1 ;     // eqn (7)(10)(12)
        beta += d2 ;
        gamma += d3 ;
 
        e1 = a - gamma - alfa ;       // eqn (5) fitting error [e1,e2,e3] ... difference of coeff
        e2 = b - alfa*gamma - beta ;
        e3 = c - gamma*beta ;

        eeee=eee ;              // ee before prior
        eee=ee ;                // prior ee
        ee=e1*e1+e2*e2+e3*e3 ;  // dot(e,e)    size of error^2

        if(ee == zero || ee == eee || ee == eeee) break ; 
    }

   
    // c--------------------- Solve Quadratic Equation ---------------------
    Solve_t cc1,diskr ;
    cc1=alfa/2. ;
    diskr=cc1*cc1-beta ;

    xx[0] = -gamma  ;  

    unsigned nr = diskr > zero ? 3 : 1 ; 
    if(nr == 3)
    { 
         diskr =sqrt(diskr) ;
         xx[1] = cc1 > 0. ? -cc1-diskr  : -cc1+diskr  ; 
         xx[2] = beta/xx[0] ;
    }
    return nr ; 
}





