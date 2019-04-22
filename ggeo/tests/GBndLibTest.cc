// ggv --bnd

#include <cassert>


#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"
#include "GVector.hh"
#include "GItemList.hh"


#include "OPTICKS_LOG.hh"


class GBndLibTest 
{
    public:
        GBndLibTest(GBndLib* blib) : m_blib(blib) {} ;
        void test_add();
    private:
        GBndLib* m_blib ; 
};

void GBndLibTest::test_add()
{
    const char* spec = "Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali" ; // omat/osur/isur/imat
    //assert(blib->contains(spec));
    bool flip = true ; 
    m_blib->add(spec, flip);

    m_blib->setBuffer(m_blib->createBuffer());
    m_blib->getBuffer()->save("$TMP/bbuf.npy");
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv) ;
    ok.configure(); 

    LOG(info) << " ok " ; 

    GBndLib* blib = GBndLib::load(&ok);

    LOG(info) << " loaded blib " ; 
    GMaterialLib* mlib = GMaterialLib::load(&ok);
    GSurfaceLib*  slib = GSurfaceLib::load(&ok);

    LOG(info) << " loaded all " 
              << " blib " << blib
              << " mlib " << mlib
              << " slib " << slib
              ;


    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);
    blib->dump();

    blib->dumpMaterialLineMap();

    assert( blib->getNames() == NULL && " expect NULL names before the close ") ; 

    blib->saveAllOverride("$TMP"); // writing to geocache in tests not allowed, as needs to work from shared install

    assert( blib->getNames() != NULL && " expect non-NULL names after the close ") ; 

    blib->dumpNames("names");

    //test_add(blib);

 
    return 0 ; 
}



