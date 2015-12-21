#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "assert.h"

#include <iostream>

void test_NumpyEvt()
{   
    NumpyEvt evt("cerenkov", "1", "dayabay") ;

    evt.setGenstepData(evt.loadGenstepFromFile());

    evt.dumpPhotonData();
}

int main()
{
    test_NumpyEvt();
    return 0 ;
}
