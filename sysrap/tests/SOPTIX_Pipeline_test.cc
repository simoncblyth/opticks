/**
SOPTIX_Pipeline_test.sh 
=======================

::
 
    ~/o/sysrap/tests/SOPTIX_Pipeline_test.sh 
    ~/o/sysrap/tests/SOPTIX_Pipeline_test.cc

Related::

    ~/o/sysrap/tests/SOPTIX_Module_test.sh

**/

#include "spath.h"
#include "scuda.h"

#include "SOPTIX.h"
#include "SOPTIX_Module.h"
#include "SOPTIX_Pipeline.h"

int main()
{
    SOPTIX opx ; 
    std::cout << opx.desc() ; 

    SOPTIX_Options opt ;  

    const char* _ptxpath = "$OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" ; 
    SOPTIX_Module mod(opx.context, opt,  _ptxpath );  
    std::cout << mod.desc() ; 

    SOPTIX_Pipeline pip(opx.context, mod.module, opt ); 
    std::cout << pip.desc() ; 

    return 0 ; 
}

