#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "CG4.hh"

#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks* m_opticks = new Opticks(argc, argv);

    m_opticks->setMode( Opticks::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    m_opticks->configure();


    OpticksEvent* m_evt = m_opticks->makeEvent();     // has to be after configure


    CG4* m_geant4 = new CG4(m_opticks) ; 

    m_geant4->setEvent(m_evt);

    m_geant4->configure();


    m_geant4->initialize();

    m_geant4->interactive(argc, argv);

    m_geant4->propagate();


    m_evt->save();

    m_geant4->cleanup();

    LOG(info) << "exiting " << argv[0] ; 

    return 0 ; 
}

