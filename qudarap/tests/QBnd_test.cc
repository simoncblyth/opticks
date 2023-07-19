/**
QBnd_test.cc
==============

Canonically built standalone with::

   ./QBnd_test.sh 



**/

#include "NP.hh"
#include "QBnd.hh"

int main()
{
    NP* bnd = NP::Load("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard/bnd.npy") ; 
    std::cout << " bnd " << ( bnd ? bnd->sstr() : "-" ) << std::endl ; 
    if(bnd == nullptr) return 1 ; 

    QBnd qb(bnd) ; 
    qb.save("$FOLD"); 

    return 0 ; 
}
