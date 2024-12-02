/**
QCurandState_Test.cc
======================

Testing the chunk-centric approach 

~/o/qudarap/tests/QCurandState_Test.sh


**/

#include <iostream>
#include "QCurandState.h"

int main(int argc, char** argv)
{
    std::cout << "[" << argv[0] << "\n" ; 

    QCurandState* cs = QCurandState::Create() ; 

    std::cout 
        << "[main cs.desc \n" 
        << cs->desc() 
        << "]main cs.desc \n" 
        ;

    std::cout << "]" << argv[0] << "\n" ; 
    return 0 ; 
}
