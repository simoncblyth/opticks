/**
SOPTIX_Module_test.sh 
=======================

::
 
    ~/o/sysrap/tests/SOPTIX_Module_test.sh 
    ~/o/sysrap/tests/SOPTIX_Module_test.cc

Related::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh

**/

#include "spath.h"
#include "scuda.h"

#include "SOPTIX.h"
#include "SOPTIX_Module.h"

int main()
{
    SOPTIX ox ; 
    std::cout << ox.desc() ; 

    SOPTIX_Options opt ;  

    const char* _ptxpath = "$OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" ; 
    // SHOULD BE: const char* _ptxpath = "$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx" ; 

    SOPTIX_Module md(ox.context, opt,  _ptxpath );  
    std::cout << md.desc() ; 

    return 0 ; 
}

