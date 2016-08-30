#include "OKG4_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKG4_LOG__ ; 

    LOG(info) << argv[0] ; 
    

    return 0 ; 
}
