#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GNodeLib.hh"

/**
GNodeLibTest
=============

See also ana/GNodeLib.py 

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    Opticks ok(argc, argv);
    ok.configure();
 
    GNodeLib* nlib = GNodeLib::Load(&ok); 
    assert(nlib);  

    LOG(info) << "nlib " << nlib ; 
    nlib->Dump("GNodeLibTest"); 
    LOG(info) << " update geocache with DayaBay geometry using geocache-dx " ; 


    return 0 ; 
}
