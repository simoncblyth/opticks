#include "NEmitConfig.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ; 

    NEmitConfig nec(NULL) ;

    nec.dump();


    return 0 ; 
}



