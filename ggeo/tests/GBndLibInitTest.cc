#include <cassert>

#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"
#include "GItemList.hh"


#include "OPTICKS_LOG.hh"

/*
#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"

*/

int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);
/*
    PLOG_(argc, argv);
    BRAP_LOG__ ;
    NPY_LOG__ ;
    GGEO_LOG__ ;
*/

    Opticks ok ;
    ok.configure();

    GMaterialLib* mlib = GMaterialLib::load(&ok);
    GSurfaceLib*  sbas = GSurfaceLib::load(&ok);
   
    if(!mlib) LOG(fatal) << " failed to load mlib " ; 
    if(!mlib) return 0 ; 
    
    if(!sbas) LOG(fatal) << " failed to load sbas : basis slib  " ; 
    if(!sbas) return 0 ; 

    GBndLib*      blib = new GBndLib(&ok) ;
    GSurfaceLib*  slib = new GSurfaceLib(&ok);

    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);

    LOG(info) << argv[0]
              << " blib " << blib
              << " mlib " << mlib
              << " sbas " << sbas
              << " slib " << slib
              ;

    blib->dump();
    blib->dumpMaterialLineMap();
    blib->saveAllOverride("$TMP/GBndLibInitTest");  // writing to geocache in tests not allowed

    return 0 ; 
}


