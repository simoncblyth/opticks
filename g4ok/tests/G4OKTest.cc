#include <cassert>
#include "OPTICKS_LOG.hh"
#include "G4Opticks.hh"

/**
G4OKTest
===============

This is aiming to replace::

   okg4/tests/OKX4Test.cc 

in order to reduce duplicated code between G4Opticks and here
and make G4Opticks functionality testable without ascending to the 
detector specific level.

**/


struct G4OKTest 
{
    G4OKTest(int argc, char** argv) 
        :
        m_g4ok(new G4Opticks)
    {
        OPTICKS_LOG(argc, argv) ;
        const char* gdmlpath = PLOG::instance->get_arg_after("--gdmlpath", NULL) ;

        m_g4ok->setGeometry(gdmlpath);  

        LOG(info) << m_g4ok->desc() ; 
        m_g4ok->doSensorDataTest("G4OKTest::G4OKTest"); 
    }

    int rc(){ return 0 ; }

    G4Opticks* m_g4ok ; 

};



int main(int argc, char** argv)
{
    G4OKTest g4okt(argc, argv); 
    return g4okt.rc() ;
}


