#include <cassert>
#include "GCache.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "NPY.hpp"

int main()
{
    GCache gc("GGEOVIEW_");

    GBndLib* blib = GBndLib::load(&gc);
    GMaterialLib* mlib = GMaterialLib::load(&gc);
    GSurfaceLib*  slib = GSurfaceLib::load(&gc);

    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);
    blib->dump();

    const char* spec = "Bialkali/Vacuum//lvPmtHemiCathodeSensorSurface" ; // imat/omat/isur/osur
    assert(blib->contains(spec));

    bool flip = true ; 
    blib->add(spec, flip);

    blib->dump();


    blib->setBuffer(blib->createBuffer());
    blib->getBuffer()->save("/tmp/bbuf.npy");


    return 0 ; 
}

