/**
QSimDescTest.cc
================

**/

#include <cuda_runtime.h>
#include "scuda.h"
#include "squad.h"

#include "QSim.hh"

int main(int argc, char** argv)
{
    std::cout << QSim::Desc() << std::endl ; 
    std::cout << QSim::Switches() << std::endl ; 
    return 0 ;
}

    


