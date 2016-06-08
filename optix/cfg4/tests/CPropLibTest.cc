// op --cproplib 
// op --cproplib 0
// op --cproplib GdDopedLS

#include "Opticks.hh"
#include "CPropLib.hh"

#include "BLog.hh"

int main(int argc, char** argv)
{
    Opticks* m_opticks = new Opticks(argc, argv, "CPropLibTest.log");
    
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    CPropLib* m_lib = new CPropLib(m_opticks); 

    m_lib->dump();

    //m_lib->dumpMaterials();






    return 0 ; 
}
