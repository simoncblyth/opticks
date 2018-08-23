// TEST=NGSTest om-t

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "NGS.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ; 
    const char* path = argc > 1 ? argv[1] : def ; 

    NPY<float>* np = NPY<float>::load(path) ; 
    if(np == NULL) return 0 ; 

    NGS* gs = new NGS(np) ; 

    unsigned modulo = 1000 ; 
    unsigned margin = 10 ;  
    gs->dump( modulo, margin ) ; 

    return 0 ; 
}
