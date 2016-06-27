// op --cproplib 
// op --cproplib 0
// op --cproplib GdDopedLS

#include "Opticks.hh"
#include "CPropLib.hh"
#include "CFG4_BODY.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    Opticks* m_opticks = new Opticks(argc, argv);
    
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    CPropLib* m_lib = new CPropLib(m_opticks); 

    m_lib->dump();

    //m_lib->dumpMaterials();



    return 0 ; 
}
