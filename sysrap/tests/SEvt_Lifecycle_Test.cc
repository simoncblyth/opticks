#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(); 
    assert(evt); 

    for(int i=0 ; i < 10 ; i++)
    {
        SEvt::AddTorchGenstep(); 
        SEvt::SetIndex(i); 



        SEvt::Save(); 
        SEvt::Clear(); 
    }
    return 0 ; 
}
