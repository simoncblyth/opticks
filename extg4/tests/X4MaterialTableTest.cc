#include "Opticks.hh"
#include "GMaterialLib.hh"
#include "X4MaterialTable.hh"
#include "X4OpNoviceMaterials.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4OpNoviceMaterials opnov ; 

    assert( opnov.water && opnov.air ) ; 

    Opticks ok(argc, argv);
    ok.configure();

    GMaterialLib* mlib = new GMaterialLib(&ok);

    X4MaterialTable::Convert(mlib) ; 

    assert( mlib->getNumMaterials() == 2 ); 

    mlib->dump();

    return 0 ; 
}
