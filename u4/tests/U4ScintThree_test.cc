// ~/o/u4/tests/U4ScintThree_test.sh

#include "spath.h"
#include "U4ScintThree.h"

int main()
{
    const char* material_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material");
    std::cout << " material_dir [" << ( material_dir ? material_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(material_dir) ;

    U4ScintThree* scint = U4ScintThree::Create(fold);
    std::cout << ( scint ? scint->desc() : "no-scint3" ) << "\n"  ;
    if(!scint) return 0 ;
    scint->save("$FOLD");

    return 0 ;
}
