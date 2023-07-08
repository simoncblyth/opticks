/**
stree_mat_test.cc
===================

The old/new fold reorganization has made this 
executable rather pointless. It just constructs 
and saves a new fold with the contents of two folds
"GGeo" and "stree/standard".

**/

#include "NPFold.h"

int main(int argc, char** argv)
{
    const char* ss = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ; 
    NPFold* gg = NPFold::Load(ss, "GGeo") ;  
    NPFold* st = NPFold::Load(ss, "stree/standard") ;  

    NPFold* fold = new NPFold ; 
    fold->add_subfold("gg", gg ); 
    fold->add_subfold("st", st ); 
    fold->save("$FOLD"); 
 
    return 0 ; 
}
