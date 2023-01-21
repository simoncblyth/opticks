
#include "OPTICKS_LOG.hh"
#include "C4.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    C4 c4 ; 

    LOG(info) << c4.desc() ; 

    LOG(info) << argv[0] ; 

    return 0 ; 
}
