#include <cassert>
#include "OKConf.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKConf::Dump("SOKConfTest") ; 

    int rc = OKConf::Check() ; 

    LOG(info) << " OKConf::Check() " << rc ;  

    assert( rc == 0 ); 

    return rc ;
}
