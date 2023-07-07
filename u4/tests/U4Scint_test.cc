// ./U4Scint_test.sh 

#include "U4Scint.h"

int main()
{
    const char* BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry" ;    
    const char* name = "LS" ; 
    NPFold* fold = NPFold::Load(BASE,"SSim/stree/material",name) ;  

    U4Scint scint(fold, name ); 
    std::cout << scint.desc() ; 

    scint.save("$FOLD"); 

    return 0 ; 
}
