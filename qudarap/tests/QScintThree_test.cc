/**

~/o/qudarap/tests/QScintThree_test.sh

**/


#include "spath.h"
#include "U4ScintThree.h"
#include "QScintThree.h"

int main()
{
    const char* material_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material");
    std::cout << " material_dir [" << ( material_dir ? material_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(material_dir) ;

    U4ScintThree* scint = U4ScintThree::Create(fold);
    std::cout << ( scint ? scint->desc() : "no-scint3" ) << "\n"  ;
    if(!scint) return 0 ;
    scint->save("$FOLD");

    QScintThree* qs = new QScintThree( scint->icdf );
    std::cout << " qs " << ( qs ? qs->desc() : "-" ) << "\n" ;

    return 0 ;
}



