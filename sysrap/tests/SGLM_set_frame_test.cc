#include "SGLM.h"

int main(int argc, char** argv)
{
    sframe fr ; 
    fr.load("$BASE") ; 

    SGLM* sglm = new SGLM  ; 
    sglm->set_frame(fr) ; 
    std::cout << sglm->desc()  ; 

    return 0 ; 
} 
