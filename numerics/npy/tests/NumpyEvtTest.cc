#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "assert.h"

void test_NumpyEvt()
{   
    NumpyEvt evt ;
    evt.setGenstepData(NPY::load("cerenkov", "1"));
}

int main()
{
    test_NumpyEvt();
    return 0 ;
}
