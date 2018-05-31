
#include "OPTICKS_LOG.hh"

#include "G4PhysicsOrderedFreeVector.hh"
#include "CVec.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    CVec* v = CVec::MakeDummy(5); 

    G4PhysicsOrderedFreeVector* vec = v->getVec() ;  

    LOG(info) << *vec  ; 

    LOG(info) << v->digest() ;


    return 0 ; 
}


