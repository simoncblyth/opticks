/**
QCurandState_Test.cc
======================

Testing the chunk-centric approach 

**/

#include <iostream>
#include "QCurandState.h"

int main()
{
    _QCurandState* cs = _QCurandState::Create() ; 
    std::cout << cs->desc() ;

    return 0 ; 
}
