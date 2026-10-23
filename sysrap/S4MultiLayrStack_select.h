#pragma once
/**
S4MultiLayrStack_select.h
==========================

Single header that picks which impl of TMM calculation to use

S4MultiLayrStack.h
    hardcodes WITH_SCOMPLEX:1 - SO IT USES LITE scomplex.h RATHER THAN thrust/complex.h

    * but observe no significant performance change - see ~/opticks/notes/optixReorder.h

**/

#ifdef WITH_CUSTOM4
#include "C4MultiLayrStack.h"
//#include "S4MultiLayrStack.h"
#endif


