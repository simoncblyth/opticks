#include "Opticks.hh"
#include "CG4.hh"

int main(int argc, char** argv)
{
    Opticks* m_opticks = new Opticks(argc, argv, "CG4Test.log");

    m_opticks->setMode( Opticks::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    CG4* m_geant4 = new CG4(m_opticks) ; 

    m_geant4->configure(argc, argv);

    m_geant4->initialize();

    m_geant4->interactive(argc, argv);

    m_geant4->propagate();

    m_geant4->save();


    return 0 ; 
}

