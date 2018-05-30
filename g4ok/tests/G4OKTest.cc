#include <cassert>
#include "G4OpticksManager.hh"


/*
// hmm every linked Opticks package should define a OPTICKS_SYSRAP OPTICKS_BRAP ... macro
// so the below includes can be automated according to what is linked 

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "MESHRAP_LOG.hh"
#include "OKGEO_LOG.hh"

// viz libs are not deps of G4OK
//#include "OGLRAP_LOG.hh"
//#include "OKGL_LOG.hh"
//#include "OK_LOG.hh"
//#include "OKG4_LOG.hh"

#include "CUDARAP_LOG.hh"
#include "THRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"

// not-viz but not yet a dep
//#include "CFG4_LOG.hh"

#include "G4OK_LOG.hh"


*/


#include "OPTICKS_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;
    MESHRAP_LOG__ ;
    OKGEO_LOG__ ;

    CUDARAP_LOG__ ;
    THRAP_LOG__ ;
    OXRAP_LOG__ ;
    OKOP_LOG__ ;

  // viz libs are not deps of G4OK  
  //  OGLRAP_LOG__ ;
  //  OKGL_LOG__ ;
  //  OK_LOG__ ;
  //  OKG4_LOG__ ;

  // neither this one, yet 
  //  CFG4_LOG__ ;

    G4OK_LOG__ ; 


    G4OpticksManager* ok = G4OpticksManager::GetOpticksManager() ; 

    assert( ok ) ; 

    return 0 ;
}
