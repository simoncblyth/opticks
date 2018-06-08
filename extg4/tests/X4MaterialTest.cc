#include "G4Material.hh"

#include "X4Material.hh"
#include "X4OpNoviceMaterials.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    X4OpNoviceMaterials opnov ; 

    G4Material* water = opnov.water ;

    GMaterial* wine = X4Material::Convert(water) ; 

    wine->Summary();

    GMaterialLib::dump(wine) ; 

    return 0 ; 
}
