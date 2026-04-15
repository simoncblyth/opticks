// ~/o/u4/tests/U4Scint_test.sh

#include "spath.h"
#include "U4Scint.h"

int main()
{
    const char* material_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material");
    std::cout << " material_dir [" << ( material_dir ? material_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(material_dir) ;

    U4Scint* scint = U4Scint::Create(fold);
    std::cout << ( scint ? scint->desc() : "no-scint" ) << "\n"  ;
    if(!scint) return 0 ;
    scint->save("$FOLD");

    return 0 ;
}
