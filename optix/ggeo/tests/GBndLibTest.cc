#include <cassert>
//#include "GCache.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "Opticks.hh"

#include "NPY.hpp"

// run this with:   ggv --bnd

int main()
{
    Opticks ok ;
    //GCache gc(&ok);

    GBndLib* blib = GBndLib::load(&ok);
    GMaterialLib* mlib = GMaterialLib::load(&ok);
    GSurfaceLib*  slib = GSurfaceLib::load(&ok);

    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);
    blib->dump();

    blib->save();             // only saves the guint4 bnd index
    blib->saveToCache();      // save float buffer too for comparison with wavelength.npy from GBoundaryLib with GBndLibTest.npy 
    blib->saveOpticalBuffer();


/*
    const char* spec = "Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali" ; // omat/osur/isur/imat
    assert(blib->contains(spec));
    bool flip = true ; 
    blib->add(spec, flip);
    blib->setBuffer(blib->createBuffer());
    blib->getBuffer()->save("/tmp/bbuf.npy");

*/

    return 0 ; 
}

