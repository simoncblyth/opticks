/**
qstateTest.cc : CPU side test of GPU side header
==================================================

::

   ./qstateTest.sh 


**/

#include "scuda.h"
#include "squad.h"
#include "qstate.h"


int main(int argc, char** argv)
{
    qstate s ; 


    //float f = uint_as_float(0u);   this method is available when using optix7 but unclear from where 

    int i0 = -101 ; 
    unsigned u0 = int_as_unsigned( i0 ); 
    int i1 = unsigned_as_int( u0 ); 
    assert( i1 == i0 ); 


    return 0 ; 
}
