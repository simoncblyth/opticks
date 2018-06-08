#include "GMaterialLib.hh"
#include "X4MaterialTable.hh"
#include "X4OpNoviceMaterials.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    X4OpNoviceMaterials opnov ; 

    assert( opnov.water && opnov.air ) ; 

    GMaterialLib* mlib = X4MaterialTable::Convert() ; 

    assert( mlib->getNumMaterials() == 2 ); 

    mlib->dump();

    return 0 ; 
}
