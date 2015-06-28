
/*
// cannot use npy- as it is being compiled against libc++ whereas CUDA/thrust needs older libstdc++ 
#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "Types.hpp"

const char* tag = "1" ;
NPY<NumpyEvt::History_t>* phis = NPY<NumpyEvt::History_t>::load("phcerenkov", tag);
Types types ; 
types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
types.dumpFlags();
phis->Summary();

*/


#include "make_sparse_histogram.h"

#include "numpy.hpp"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include "Flags.hh"

#include <vector>
#include <iostream>
#include <fstream>


int main(int argc, char** argv)
{
    const char* fpath = "/tmp/GFlagIndexLocal.ini";
    Flags flags ;
    flags.read(fpath);
    //flags.dump();

    unsigned long long seq = 0xfedcba9876543210 ; 
    std::string sseq = flags.getSequenceString(seq);

    const char* path = "/usr/local/env/phcerenkov/1.npy" ;

    std::vector<int> shape ;
    std::vector<History_t> content ;
    aoba::LoadArrayFromNumpy<History_t>(path, shape, content );

    std::cout <<  "data size " << content.size() << std::endl ;


    unsigned int numElements = content.size();
    History_t* data = content.data();   

    make_sparse_histogram( data , numElements, &flags );

    return 0 ;
}

