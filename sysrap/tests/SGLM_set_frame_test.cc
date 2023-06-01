#include "SGLM.h"

int main(int argc, char** argv)
{
    sframe fr ; 

    fr.load("$BASE") ; 

    SGLM* sglm = new SGLM  ; 

    sglm->addlog("CSGOptiX::init", "start");

    sglm->set_frame(fr) ; 

    sglm->addlog("CSGOptiX::render_snap", "from SGLM_set_frame_test.cc" );

    sglm->writeDesc("$BASE", "SGLM_set_frame_test.log"); 

    return 0 ; 
} 
