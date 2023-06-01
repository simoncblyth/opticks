#include "SGLM.h"
#include "NP.hh"

int main(int argc, char** argv)
{
    sframe fr ; 
    fr.load("$BASE") ; 

    SGLM* sglm = new SGLM  ; 
    sglm->set_frame(fr) ; 
    sglm->writeDesc("$BASE", "SGLM_set_frame_test.log"); 

    return 0 ; 
} 
