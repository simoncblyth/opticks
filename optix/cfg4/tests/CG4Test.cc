#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "CG4.hh"

#include "PLOG.hh"
#include "CFG4_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    CFG4_LOG__ ; 


    LOG(info) << argv[0] ;

    Opticks* m_opticks = new Opticks(argc, argv);

    m_opticks->setMode( Opticks::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    m_opticks->configure();


    OpticksEvent* m_evt = m_opticks->makeEvent();     // has to be after configure


    CG4* m_geant4 = new CG4(m_opticks) ; 

    LOG(info) << "  CG4 ctor DONE "  ;

    m_geant4->setEvent(m_evt);

    m_geant4->configure();

    LOG(info) << "  CG4 configure DONE "  ;

    m_geant4->initialize();

    LOG(info) << "  CG4 initialize DONE "  ;


    m_geant4->interactive(argc, argv);

    LOG(info) << "  CG4 interactive DONE "  ;

    m_geant4->propagate();

    LOG(info) << "  CG4 propagate DONE "  ;

    m_evt->save();

    LOG(info) << "  evt save DONE "  ;

    m_geant4->cleanup();

    LOG(info) << "exiting " << argv[0] ; 

    return 0 ; 
}

