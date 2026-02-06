#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SProcessHits_EPH.h"
#include "NPFold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEvt* evt = SEvt::Create(SEvt::EGPU);   // instanciation may load input_photons
    assert(evt);

    std::cout << SEvt::DescINSTANCE() ;


    bool ip = SEvt::HasInputPhoton(SEvt::EGPU) ;

#ifdef WITH_OLD_FRAME
    sframe fr = sframe::Fabricate(0.f,0.f,1000.f) ;
    evt->setFrame(fr);
#else
    sfr fr = sfr::MakeFromTranslateExtent<float>(0.f,0.f,1000.f,2000.f);
    evt->setFr(fr);
#endif


    for(int i=0 ; i < 3 ; i++)
    {
        if(!ip) SEvt::AddTorchGenstep();

        evt->beginOfEvent(i);  // setIndex and does genstep setup for input photons
        assert( SEvt::Get(0) == evt );

        int npc = SEvt::GetNumPhotonCollected(0) ;
        for(int j=0 ; j < npc ; j++)
        {
            int track_id = j ;
            spho label = spho::Fabricate(track_id);
            evt->beginPhoton(label);


            sctx& ctx = evt->current_ctx ;

            ctx.p.set_flag(BOUNDARY_TRANSMIT);
            evt->pointPhoton(label);

            ctx.p.set_flag(BOUNDARY_TRANSMIT);
            evt->pointPhoton(label);

            ctx.p.set_flag(BOUNDARY_TRANSMIT);
            evt->pointPhoton(label);

            evt->addProcessHitsStamp(0);
            evt->addProcessHitsStamp(1);

            evt->addProcessHitsStamp(0);
            evt->addProcessHitsStamp(1);

            evt->addProcessHitsStamp(1);


            // currently needed to avoid consistent_flag assert in SEvt::finalPhoton
            ctx.p.set_flag(SURFACE_DETECT);
            label.set_eph(EPH::SAVENORM);

            evt->pointPhoton(label);

            evt->finalPhoton(label) ;  // sctx::end copies {seq,sup} into sevent arrays
        }

        std::cout << evt->descVec() << std::endl ;


        // see QSim::simulate for the needed incantation
        evt->gather();
        evt->topfold->concat();
        evt->topfold->clear_subfold();


        evt->endOfEvent(i);

        assert( SEvt::Get(0) == evt );
    }

    std::cout << evt->descDbg() ;

    return 0 ;
}
