// ~/o/u4/tests/U4ScintThreeExtra_test.sh

#include "spath.h"
#include "U4ScintThreeExtra.h"

int main()
{
    const char* ls_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material/LS");
    std::cout << " ls_dir [" << ( ls_dir ? ls_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(ls_dir) ;
    const char* name = "LS" ;

    U4ScintThreeExtra* scint = new U4ScintThreeExtra(fold, name);
    std::cout << ( scint ? scint->desc() : "-" ) << "\n"  ;
    scint->save("$FOLD");

    return 0 ;
}
