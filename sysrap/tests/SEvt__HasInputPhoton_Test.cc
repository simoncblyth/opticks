#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(); 
    assert(evt); 

    bool ip = SEvt::HasInputPhoton() ; 

    LOG(info) << "SEvt::HasInputPhoton " << ip ; 
    LOG(info) << SEvt::DescInputPhoton() ;  
 
    return 0 ; 
}
