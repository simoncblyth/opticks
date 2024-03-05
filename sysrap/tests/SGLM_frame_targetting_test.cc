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
    sframe a ; 
    sframe b ; 

    a.ce = make_float4( 0.f, 0.f, 0.f,  100.f ); 
    b.ce = make_float4( 0.f, 0.f, 0.f,  200.f ); 

    SGLM A ; 
    SGLM B ; 

    A.set_frame(a) ; 
    B.set_frame(b) ; 

    A.writeDesc("$FOLD", "A" ); 
    B.writeDesc("$FOLD", "B" ); 

    const float* A_world2clip = (const float*)glm::value_ptr(A.world2clip) ;
    const float* B_world2clip = (const float*)glm::value_ptr(B.world2clip) ;


    std::cout 
        << "A_world2clip"
        << std::endl 
        << stra<float>::Desc(A_world2clip, 16) 
        << std::endl
        << "B_world2clip"
        << std::endl 
        << stra<float>::Desc(B_world2clip, 16) 
        << std::endl
        ;
   
    
    std::cout 
        << "A.desc_world2clip_ce_corners" 
        << std::endl 
        << A.desc_world2clip_ce_corners()
        << std::endl 
        << "B.desc_world2clip_ce_corners" 
        << std::endl 
        << B.desc_world2clip_ce_corners()
        << std::endl 
        ; 


    return 0 ; 
} 
