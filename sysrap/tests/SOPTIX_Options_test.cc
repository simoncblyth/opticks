/**
SOPTIX_Options_test.cc
=======================

::

    ~/o/sysrap/tests/SOPTIX_Options_test.sh
    ~/o/sysrap/tests/SOPTIX_Options_test.cc

**/

#include <optix.h>

#include "SOPTIX_Options.h"

int main()
{
    SOPTIX_Options opt ; 
    std::cout << opt.desc() ; 

    return 0 ;
}
