/**
QCurandStateTest.cc
======================

~/o/qudarap/tests/QCurandStateTest.sh

Used at install time by::

     qudarap-prepare-installation 


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
