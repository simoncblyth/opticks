#include "OKG4Mgr.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    OKG4Mgr okg4(argc, argv);
    okg4.propagate();
    okg4.visualize();   

    return okg4.rc() ;
}
