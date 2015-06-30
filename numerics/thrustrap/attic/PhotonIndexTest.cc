
#ifdef SPARTAN
// cannot use npy- as it is being compiled against libc++ 
// whereas CUDA/thrust prior to CUDA 7.0 needs older libstdc++ 
#include "numpy.hpp"
#include "Flags.hh"

#else

#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "Types.hpp"

#endif


#include "make_sparse_histogram.h"

#include "assert.h"
#include "stdio.h"
#include "stdlib.h"

#include <vector>
#include <iostream>
#include <fstream>


int main(int argc, char** argv)
{
    unsigned long long seqtest = 0xfedcba9876543210 ; 
#ifdef SPARTAN 
    const char* fpath = "/tmp/GFlagIndexLocal.ini";
    Flags flags ;
    flags.read(fpath);
    flags.dump();
    std::string sseq = flags.getSequenceString(seqtest);
    std::cout << "sseq " << sseq << std::endl ; 
 
#else
    Types types ; 
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    types.dumpFlags();

    std::string sseq = types.getSequenceString(seqtest);
    std::cout << "sseq " << sseq << std::endl ; 
#endif



#ifdef SPARTAN
    {
        const char* path = "/usr/local/env/phcerenkov/1.npy" ;
        std::vector<int> shape ;
        std::vector<History_t> content ;
        aoba::LoadArrayFromNumpy<History_t>(path, shape, content );
        std::cout <<  "data size " << content.size() << std::endl ;

        make_sparse_histogram( content.data() , content.size(), &flags );
    }
#else
    {
        NPY<NumpyEvt::History_t>* phis = NPY<NumpyEvt::History_t>::load("phcerenkov", "1");
        phis->Summary();
        make_sparse_histogram( phis->getValues(), phis->getShape(0), &types );
    }
#endif


    return 0 ;
}

