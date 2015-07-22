#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "assert.h"

#include <iostream>

void test_NumpyEvt()
{   
    NumpyEvt evt ;
    evt.setGenstepData(NPY<float>::load("cerenkov", "1", "dayabay"));
    evt.dumpPhotonData();
}

int main()
{
    test_NumpyEvt();
    return 0 ;
}
