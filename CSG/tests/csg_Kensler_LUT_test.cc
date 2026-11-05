/**
csg_Kensler_LUT_test.cc
========================

~/o/CSG/tests/csg_Kensler_LUT_test.sh

**/

#include "NPFold.h"
#include "OpticksCSG.h"
#include "csg_classify.h"
#include "csg_Kensler_LUT.h"

int main()
{
    int ni = 2 ; 
    int nj = 3 ;
    int nk = 3 ;
    int nl = 3 ;

    NP* a = NP::Make<int>(ni, nj, nk, nl );
    int* aa = a->values<int>();

    NP* b = NP::Make<int>(ni, nj, nk, nl );
    int* bb = b->values<int>();

    NPFold* f = new NPFold ; 
    f->add("a", a );
    f->add("b", b );

    LUT lut ;

    for(int i=0 ; i < ni ; i++ )
    for(int j=0 ; j < nj ; j++ )
    for(int k=0 ; k < nk ; k++ )
    for(int l=0 ; l < nl ; l++ )
    {

        bool A_Closer = i == 0 ; 
        OpticksCSG_t operation = (OpticksCSG_t)(1+j) ;   // CSG_UNION, CSG_INTERSECTION, CSG_DIFFERENCE
        IntersectionState_t state_A = (IntersectionState_t)k ;  
        IntersectionState_t state_B = (IntersectionState_t)l ;

        int kensler_ctrl= A_Closer ? 
                            Kensler_LUT::lookup<true>(  operation , state_A, state_B )
                        :
                            Kensler_LUT::lookup<false>( operation , state_A, state_B )
                        ;

        int idx = i*nj*nk*nl + j*nk*nl + k*nl + l ;
        aa[idx] = kensler_ctrl ;



        OpticksCSG_t typecode = operation ;
        IntersectionState_t l_state = state_A ;
        IntersectionState_t r_state = state_B ;
        bool leftIsCloser = A_Closer ;
        int old_ctrl = lut.lookup( typecode , l_state, r_state, leftIsCloser ) ;
        bb[idx] = old_ctrl ;

        assert( kensler_ctrl == old_ctrl );

    }
    f->save("$FOLD");
    return 0 ;
}
