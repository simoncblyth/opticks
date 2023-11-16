#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "QEvent.hh"

struct QEvent_Lifecycle_Test
{
    static void Test(); 
}; 

/**
QEvent_Lifecycle_Test::Test
----------------------------

Comments are from an input photon centric viewpoint 
as those are useful for debugging. 

**/

void QEvent_Lifecycle_Test::Test()
{
    SEvt* sev = SEvt::Create(SEvt::EGPU) ; 
    // instanciation may load input_photons if configured
    assert( sev );  

    sev->setFramePlaceholder();       
    // calls SEvt::setFrame which 
    // for non-placeholder frame might transform the input photons
    // using the frame transform 

    QEvent* event = new QEvent ; // grabs SEvt::EGPU  

    for(int i = 0 ; i < 10 ; i++)
    { 
        // follow pattern of QSim::simulate

        int eventID = i ; 

        sev->beginOfEvent(eventID);  
        // SEvt::beginOfEvent calls SEvt::setFrameGenstep which creates 
        // the input photon genstep and calls SEvt::addGenstep

        int rc = event->setGenstep(); 
        assert( rc == 0 );
        // QEvent::setGenstep 
        //    1. calls SEvt::gatherGenstep anduploads 
        //    2. calls QEvent::setInputPhoton which invokes SEvt::gatherInputPhoton and uploads 
        //


        // IN REALITY THE LAUNCH WOULD BE HERE


        sev->endOfEvent(eventID); 
        // invoked SEvt::save and SEvt::clear_except("hit")

    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    /**
    HMM: might expect that using the lower level ssys::setenvvar 
    would allow a setting that can be overriden from the script 
    QEvent_Lifecycle_Test.sh : but that dont work for 
    OPTICKS_INPUT_PHOTON as the SEventConfig statics run before the main does. 

    Override from script  does work for GEOM because that is used as a plain envvar  
    **/

    bool overwrite = false ; 
    //ssys::setenvvar("OPTICKS_INPUT_PHOTON", "RainXZ_Z230_10k_f8.npy", overwrite ); 
    SEventConfig::SetInputPhoton("RainXZ_Z230_10k_f8.npy"); 

    ssys::setenvvar("GEOM", "TEST_CC", overwrite ); 

    QEvent_Lifecycle_Test::Test() ;
    return 0 ; 
}
