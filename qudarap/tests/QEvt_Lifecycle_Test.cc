/**
QEvt_Lifecycle_Test.cc
=========================

~/o/qudarap/tests/QEvt_Lifecycle_Test.sh


**/

#include <csignal>
#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "QEvt.hh"

struct QEvt_Lifecycle_Test
{
    static int EventLoop();
};

/**
QEvt_Lifecycle_Test::EventLoop
-------------------------------------

Comments are from an input photon centric viewpoint
as those are useful for debugging.

**/

int QEvt_Lifecycle_Test::EventLoop()
{
    SEvt* sev = SEvt::Create_EGPU() ;
    // instanciation may load input_photons if configured
    assert( sev );

    sev->setFramePlaceholder();
    // calls SEvt::setFrame which
    // for non-placeholder frame might transform the input photons
    // using the frame transform



    QEvt* event = new QEvt ; // grabs SEvt::EGPU

    int num_event = SEventConfig::NumEvent() ;
    std::cout << " num_event " << num_event << std::endl ;

    for(int i = 0 ; i < num_event ; i++)
    {
        //std::cout << i << std::endl ;
        // follow pattern of QSim::simulate

        int eventID = i ;

        sev->beginOfEvent(eventID);
        // SEvt::beginOfEvent calls SEvt::setFrameGenstep which creates
        // the input photon genstep and calls SEvt::addGenstep

        NP* igs = sev->makeGenstepArrayFromVector();
        int rc = event->setGenstepUpload_NP(igs);
        assert( rc == 0 );
        if(rc!=0) std::raise(SIGINT);

        // IN REALITY THE LAUNCH WOULD BE HERE
        // propagating the photons, changing GPU side buffers


        sev->endOfEvent(eventID);
    }
    return 0 ;
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    /**
    HMM: might expect that using the lower level ssys::setenvvar
    would allow a setting that can be overriden from the script
    QEvt_Lifecycle_Test.sh : but that dont work for
    OPTICKS_INPUT_PHOTON as the SEventConfig statics run before the main does.

    Override from script  does work for GEOM because that is used as a plain envvar
    **/

    bool overwrite = false ;
    //ssys::setenvvar("OPTICKS_INPUT_PHOTON", "RainXZ_Z230_10k_f8.npy", overwrite );
    SEventConfig::SetInputPhoton("RainXZ_Z230_10k_f8.npy");

    ssys::setenvvar("GEOM", "TEST_CC", overwrite );

    return QEvt_Lifecycle_Test::EventLoop() ;
}
