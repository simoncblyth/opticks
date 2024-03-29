#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt::Create(0); 

    bool ip = SEvt::HasInputPhoton(0) ; 

    LOG(info) << "SEvt::HasInputPhoton " << ip ; 
    LOG(info) << SEvt::DescInputPhoton(0) ;  
 
    return 0 ; 
}
