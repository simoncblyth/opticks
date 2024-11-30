/**
QCurandState_Test.cc
======================

Testing the chunk-centric approach 

~/o/qudarap/tests/QCurandState_Test.sh


**/

#include <iostream>
#include "QCurandState.h"

int main()
{
    _QCurandState* cs = _QCurandState::Create() ; 
    std::cout << cs->desc() ;

    return 0 ; 
}
