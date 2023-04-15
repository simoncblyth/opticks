#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(); 
    assert(evt); 

    for(int i=0 ; i < 3 ; i++)
    {
        // SEvt::AddTorchGenstep(); 
        SEvt::SetIndex(i); 
        assert( SEvt::Get() == evt ); 

        std::cout << evt->descVec() << std::endl ; 

        SEvt::Save(); 
        SEvt::Clear(); 
        assert( SEvt::Get() == evt ); 
    }

    std::cout << evt->descDbg() ; 

    return 0 ; 
}
