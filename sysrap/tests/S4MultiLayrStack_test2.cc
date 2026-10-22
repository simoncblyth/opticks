/**
S4MultiLayrStack_test2.cc
===========================

Adapted from ~/customgeant4/tests/C4MultiLayrStack_Test2.cc


HMM: scanning dot_pol_cross_mom_nrm needs to be consistent with minus_cos_theta
as minus_cos_theta is dot(mom,nrm) which is related to cross(mom,nrm)

HMM: simpler to think in terms of SF which is what dot_pol_cross_mom_nrm is
used to calculate in anycase      

**/
#include <cmath>
#include "S4MultiLayrStack.h"
#include "NPFold.h"
#include "sarray.h"


struct S4MultiLayrStack_test2
{
    template<typename F> static NP* aoi(    const F* ss);
    template<typename F> static NP* aoi_pol(const F* ss);

    static int main();
};



int S4MultiLayrStack_test2::main()
{
    double ss[16] = 
      {
        1.482,     0,      0,   0,
        1.920,     0,  36.49,   0,
        2.429, 1.366,  21.13,   0,
        1.000,     0,      0,   0  
      }; 

    
    NPFold* f = new NPFold ; 
    f->add( "aoi",     aoi<double>(ss) ) ; 
    f->add( "aoi_pol", aoi_pol<double>(ss) ) ; 
    f->save("$FOLD"); 
    return 0 ;
}


template<typename F>
NP* S4MultiLayrStack_test2::aoi(const F* ss )
{
    std::cout << "[aoi" << std::endl ; 

    int ni = 180 ; // aoi scan 
    int nj = 2 ;   // payload groups 
    int nk = 4 ; 
    int nl = 4 ; 

    NP* a = NP::Make<F>( ni,nj,nk,nl ); 
    F* aa = a->values<F>(); 
    F wl = 440. ; 

    for(int i=0 ; i < ni ; i++)
    {
        F frac_pi = F(i)/F(ni-1) ;   
        F theta = frac_pi*M_PI ;
        F minus_cos_theta = -cos(theta) ; 
        F dot_pol_cross_mom_nrm = 0. ; 
     
        Stack<F,4> stack  ; 
        stack.calc( wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u ); 

        //std::cout << stack.art ; 

        for(int j=0 ; j < nj ; j++)
        {
            int idx = i*nj*nk*nl + j*nk*nl ; 
            switch(j)
            {
                case 0: memcpy( aa+idx , ss               , sizeof(F)*nk*nl ) ; break ; 
                case 1: memcpy( aa+idx , stack.art.cdata(), sizeof(F)*nk*nl ) ; break ; 
            }
        }
    }
    std::cout << "]aoi" << std::endl ; 
    return a ; 
}


/**
aoi_pol
--------

Scanning across 180 degrees of angle of incidence and 
360 degrees of polarization direction. 

Hmm need to pick lots of mom vectors at different 
angle of incidence. But there is rotational symmetry here 
so keep inside the Y=0 plane for simplicity.  

              Z
             nrm 
              :
          +   :
           \  :
        mom \ : 
             \:
    ----------+-----------  X


  
                                        
    | a_x |   | b_x |       |   i    j    k   |       |   a_y b_z - a_z b_y   | 
    |     |   |     |       |                 |       |                       |
    | a_y | X | b_y |   =   |  a_x  a_y  a_z  |   =   | -(a_x b_z - a_z b_x)  |
    |     |   |     |       |                 |       |                       |
    | a_z |   | b_z |       |  b_x  b_y  b_z  |       |   a_x b_y - a_y b_x   |

    https://en.wikipedia.org/wiki/Cross_product

    

    cross_mom_nrm 

     | st  |    | 0 |      |   i    j    k    |       |       0         |    |  0   |
     |     |    |   |      |                  |       |                 |    |      |
     |  0  | X  | 0 |  =   |  st    0    mct  |   =   |  -(st - mct.0 ) | =  | -st  |
     |     |    |   |      |                  |       |                 |    |      |
     | mct |    | 1 |      |   0    0    1    |       |       0         |    |  0   |

**/

template<typename F>
NP* S4MultiLayrStack_test2::aoi_pol(const F* ss )
{
    std::cout << "[aoi_pol" << std::endl ; 
    F wl = 440. ; 
    std::array<F,3> nrm = {{ 0. , 0., 1. }}  ; 

    int ni = 180 ; // aoi scan 
    int nj = 360 ; // pol scan  
    int nk = 2 ;   // payload groups 
    int nl = 4 ; 
    int nm = 4 ; 

    NP* a = NP::Make<F>( ni,nj,nk,nl,nm ); 
    F* aa = a->values<F>(); 

    for(int i=0 ; i < ni ; i++)
    {
        F frac_pi = F(i)/F(ni-1) ;   
        F theta = frac_pi*M_PI ;
        F sint = sin(theta) ; 
        F cost = cos(theta) ; 
 
        // fix mom within XZ plane 
        std::array<F,3> mom = {{ sint , 0., -cost }}  ; 
        F minus_cos_theta = V::dot<F>( mom, nrm ) ; 

        std::array<F,3> cross_mom_nrm = {{ 0., 0., 0. }};
        V::cross<F>( cross_mom_nrm, mom, nrm );    
        // get vector transverse to plane of incidence

        for(int j=0 ; j < nj ; j++)
        {
            F frac_twopi = F(j)/F(nj-1) ;  

            std::array<F,3> pol = {{ 0., 0., 0. }}  ; 
            V::make_transverse<F>( pol, mom, frac_twopi ); 
            assert( std::abs( V::length<F>(pol) - 1. ) < 1e-7 );  

            F dot_pol_mom = V::dot<F>( pol, mom ); 
            assert( dot_pol_mom < 1e-7 ); 

            F dot_pol_cross_mom_nrm = V::dot<F>( pol, cross_mom_nrm ); 
            F expect_dot_pol_cross_mom_nrm = -sint*pol[V::Y] ; 
            F diff_dot_pol_cross_mom_nrm = std::abs( expect_dot_pol_cross_mom_nrm - dot_pol_cross_mom_nrm ) ;
            assert( diff_dot_pol_cross_mom_nrm < 1e-7 ) ; 

            Stack<F,4> stack  ; 
            stack.calc( wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u ); 

            if(stack.art.SF > 1.) 
            {
                std::cout 
                << " i " << std::setw(3) << i 
                << " j " << std::setw(3) << j
                << std::endl 
                << " nrm                          :" << V::desc<F>( nrm ) << std::endl 
                << " mom                          :" << V::desc<F>( mom ) << std::endl 
                << " cross_mom_nrm                :" << V::desc<F>( cross_mom_nrm ) << std::endl   
                << " minus_cos_theta              :" << std::fixed << std::setw(10) << std::setprecision(4) << minus_cos_theta << std::endl 
                << " pol                          :" << V::desc<F>( pol ) << std::endl 
                << " dot_pol_mom                  :" << std::fixed << std::setw(10) << std::setprecision(4) << dot_pol_mom << std::endl 
                << " dot_pol_cross_mom_nrm        :" << std::fixed << std::setw(10) << std::setprecision(4) << dot_pol_cross_mom_nrm << std::endl 
                << " expect_dot_pol_cross_mom_nrm :" << std::fixed << std::setw(10) << std::setprecision(4) << expect_dot_pol_cross_mom_nrm << std::endl 
                << " diff_dot_pol_cross_mom_nrm   :" << std::fixed << std::setw(10) << std::setprecision(4) << diff_dot_pol_cross_mom_nrm << std::endl 
                << " stack.art.SF                 :" << std::fixed << std::setw(10) << std::setprecision(4) << stack.art.SF << std::endl 
                << std::endl 
                << stack.art
                ;
            }
            assert( stack.art.SF <= 1. ); 

            for(int k=0 ; k < nk ; k++)
            {
                int idx = i*nj*nk*nl*nm + j*nk*nl*nm + k*nl*nm ; 
                switch(k)
                {
                    case 0: memcpy( aa+idx , ss               , sizeof(F)*nl*nm ) ; break ; 
                    case 1: memcpy( aa+idx , stack.art.cdata(), sizeof(F)*nl*nm ) ; break ; 
                }
            }
        }
    }
    std::cout << "]aoi_pol" << std::endl ; 
    return a ; 
}

int main()
{
    return S4MultiLayrStack_test2::main() ; 
}


