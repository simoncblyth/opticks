/**
SGLM_set_frame_test.cc
=======================

1. load sframe from $SFRAME_FOLD/sframe.npy
2. instanciate SGLM
3. set the frame into SGLM 
4. write SGLM description to $SFRAME_FOLD/SGLM_set_frame_test.log

::

    ~/o/sysrap/tests/SGLM_set_frame_test.sh info_build_run_cat

**/

#include "SGLM.h"

int main(int argc, char** argv)
{
    sframe fr = sframe::Load("$SFRAME_FOLD") ; 

    std::cout << "//SGLM_set_frame_test.main load sframe from SFRAME_FOLD " << std::endl ; 

    SGLM* sglm = new SGLM  ; 

    sglm->addlog("CSGOptiX::init", "start");

    sglm->set_frame(fr) ; 

    sglm->addlog("CSGOptiX::render_snap", "from SGLM_set_frame_test.cc" );

    sglm->writeDesc("$SFRAME_FOLD", "SGLM_set_frame_test", ".log" ); 

    std::cout << "//SGLM_set_frame_test.main write frame description to SFRAME_FOLD/SGLM_set_frame_test.log" << std::endl ; 

    return 0 ; 
} 
