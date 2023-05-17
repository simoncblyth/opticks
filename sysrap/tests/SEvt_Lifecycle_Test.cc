#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(); 
    assert(evt); 

    bool ip = SEvt::HasInputPhoton() ; 
    sframe fr = sframe::Fabricate(0.f,0.f,1000.f) ; 
    evt->setFrame(fr); 

    for(int i=0 ; i < 3 ; i++)
    {
        if(!ip) SEvt::AddTorchGenstep(); 

        SEvt::BeginOfEvent(i);  // setIndex and does genstep setup for input photons
        assert( SEvt::Get() == evt ); 

        int npc = SEvt::GetNumPhotonCollected() ; 
        for(int j=0 ; j < npc ; j++)
        { 
            int track_id = j ; 
            spho label = spho::Fabricate(track_id);  
            evt->beginPhoton(label);  

            evt->pointPhoton(label); 
            evt->pointPhoton(label); 
            evt->pointPhoton(label); 

            evt->addProcessHitsStamp(); 
            evt->addProcessHitsStamp(); 
            evt->addProcessHitsStamp(); 

            evt->pointPhoton(label); 

            evt->finalPhoton(label) ;  // sctx::end copies {seq,sup} into sevent arrays 
        }

        std::cout << evt->descVec() << std::endl ; 

        SEvt::Save(); 
        SEvt::Clear(); 
        assert( SEvt::Get() == evt ); 
    }

    std::cout << evt->descDbg() ; 

    return 0 ; 
}
