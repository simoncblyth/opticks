
// clang Vecgeom_Solve.cc -lc++ && ./a.out && rm a.out

            /*
            In [45]: collect(expand( (x-x0)*(x-x1)*(x-x2) ),x)
            Out[45]: x**3 + x**2*(-x0 - x1 - x2) + x*(x0*x1 + x0*x2 + x1*x2) - x0*x1*x2
            */



#include <cassert>
#include <cmath>

#include "fascending.h"
#include "Vecgeom_Solve.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>


struct CubicTest
{
    unsigned num  ; 
    unsigned nr  ; 
    unsigned msk ;
    bool coeff ; 
 
    float r0[3] ; 
    float co[3] ; 
    float r1[3] ; 

    std::string desc_arr(const float* a, unsigned n) const 
    {
         std::stringstream ss ; 
         for(unsigned i=0 ; i < num ; i++) 
         {
             if( i < n )
             {
                 ss << std::setw(10) << a[i] << " " ; 
             }
             else
             {
                 ss << std::setw(10) << "-" << " " ; 
             }
         }
         return ss.str();
    }

    void dump()
    {
        std::cout 
            << " nr " << nr 
            << " r0 " << desc_arr(r0, coeff ? 0 : num) 
            << " r1 " << desc_arr(r1, nr  )
            << " co " << desc_arr(co, num )
            << " " << SolveTypeMask(msk)
            << std::endl 
            ;
    }

    CubicTest(float x0, float x1, float x2, unsigned msk_, bool coeff_ ) : num(3), nr(0), msk(msk_), coeff(coeff_) 
    {
        if(coeff)
        {
            r0[0] = 0 ; 
            r0[1] = 0 ; 
            r0[2] = 0 ; 
     
            co[0] = x0 ; 
            co[1] = x1 ; 
            co[2] = x2 ; 
        } 
        else
        {
            r0[0] = x0 ; 
            r0[1] = x1 ; 
            r0[2] = x2 ; 
            fascending_ptr( 3, r0 );

            co[0] = -x0*x1*x2 ;
            co[1] =  x0*x1 + x0*x2 + x1*x2 ;
            co[2] = -x0 - x1 - x2 ;
        }

        nr = SolveCubic( co[2],co[1],co[0], r1, msk );
        fascending_ptr( nr, r1 );

        dump();
    }
};



int main()
{
    static const unsigned N = 3 ; 

    unsigned msk[N] ; 
    msk[0] = SOLVE_VECGEOM ;
    msk[1] = SOLVE_UNOBFUSCATED  ;
    msk[2] = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD ;


    //   a0 + a1*x + a2*x**2 + 1*x**3 
  
    for(unsigned i=0 ; i < N ; i++) CubicTest t(0,0,0, msk[i], true) ;   // x**3 = 0
    for(unsigned i=0 ; i < N ; i++) CubicTest t(-1,0,0, msk[i], true) ;   // x**3 = 1  



    float x0 = 1 ; 
    float x1 = 2 ; 
    float x2 = 3 ;  

    for(float s=1 ; s < 1000.f ; s+=100. )
    {
        for(unsigned i=0 ; i < N ; i++) CubicTest t(x0*s,x1*s,x2*s, msk[i], false) ; 
    }
    for(float s=1 ; s < 1000.f ; s+=100. )
    {
        for(unsigned i=0 ; i < N ; i++) CubicTest t(x0*s,x1*s,x2*s*100., msk[i], false) ; 
    }




    return 0 ; 
}


