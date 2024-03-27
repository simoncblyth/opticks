/**
SGLM_frame_targetting_test.cc
===============================

::

    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh info_build_run


1. fabricate frames A and B in code, with known simple relationhip like double the extent
2. instanciate two correponding SGLM instances from the two frames
3. write the two SGLM descriptions 
4. compare the two descriptions

**/

#include "SGLM.h"

int main(int argc, char** argv)
{
    sfr a ; 
    a.set_extent(100.f); 

    SGLM A ; 
    A.set_frame(a) ; 
    A.writeDesc("$FOLD", "A" ); 

    std::cout 
        << "A_MVP (aka world2clip) "
        << std::endl 
        << stra<float>::Desc(A.MVP_ptr, 16) 
        << std::endl
        << "A.desc_MVP_ce_corners" 
        << std::endl 
        << A.desc_MVP_ce_corners()
        << std::endl 
        ;


    sfr b ; 
    b.set_extent(200.f); 

    SGLM B ; 
    B.set_frame(b) ; 
    B.writeDesc("$FOLD", "B" ); 

    std::cout 
        << "B_MVP (aka world2clip) "
        << std::endl 
        << stra<float>::Desc(B.MVP_ptr, 16) 
        << std::endl
        << "B.desc_MVP_ce_corners" 
        << std::endl 
        << B.desc_MVP_ce_corners()
        << std::endl 
        ; 

    return 0 ; 
} 
