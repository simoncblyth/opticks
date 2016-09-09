#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksEvent.hh"
#include "OpticksHub.hh"
#include "OpticksRun.hh"

#include "CG4.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);

    NPY_LOG__ ; 
    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    CFG4_LOG__ ; 


    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    OpticksRun* run = hub.getRun();
    
    CG4* g4 = new CG4(&hub) ; 

    LOG(info) << "  CG4 ctor DONE "  ;

    g4->configure();

    LOG(info) << "  CG4 configure DONE "  ;

    g4->initialize();

    LOG(info) << "  CG4 initialize DONE "  ;


    g4->interactive();

    LOG(info) << "  CG4 interactive DONE "  ;


    run->createEvent();

    g4->propagate();

    LOG(info) << "  CG4 propagate DONE "  ;


    OpticksEvent* evt = run->getG4Event();
    assert(evt->isG4()); 
    evt->save();

    LOG(info) << "  evt save DONE "  ;

    g4->cleanup();

    LOG(info) << "exiting " << argv[0] ; 

    return 0 ; 
}

