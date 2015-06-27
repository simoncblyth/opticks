
/*
#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "Types.hpp"
*/


#include "make_sparse_histogram.h"

#include "numpy.hpp"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"

#include <vector>
#include <iostream>



int main(int argc, char** argv)
{
    /*
    const char* tag = "1" ;
    NPY<NumpyEvt::History_t>* phis = NPY<NumpyEvt::History_t>::load("phcerenkov", tag);
    Types types ; 
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    types.dumpFlags();
    phis->Summary();
    */

    typedef unsigned long History_t ; 
    const char* path = "/usr/local/env/phcerenkov/1.npy" ;

    std::vector<int> shape ;
    std::vector<History_t> content ;
    aoba::LoadArrayFromNumpy<History_t>(path, shape, content );

    std::cout <<  "data size " << content.size() << std::endl ;


    unsigned int numElements = content.size();
    History_t* data = content.data();   

    make_sparse_histogram( data , numElements );

    return 0 ;
}

