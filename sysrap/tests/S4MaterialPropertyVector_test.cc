#include <vector>
#include <iostream>

#include "S4MaterialPropertyVector.h"

int main()
{
    std::vector<G4MaterialPropertyVector*> vv ; 
    vv.resize(3) ; 

    G4MaterialPropertyVector* v0 = S4MaterialPropertyVector::Make_V(10.) ; 
    G4MaterialPropertyVector* v1 = S4MaterialPropertyVector::Make_V(20.) ; 
    G4MaterialPropertyVector* v2 = S4MaterialPropertyVector::Make_V(30.) ; 

    vv[0] = v0 ; 
    vv[1] = v1 ; 
    vv[2] = v2 ; 

    NPFold* fold = S4MaterialPropertyVector::Serialize_VV(vv) ; 

    std::cout << fold->desc() ; 

    std::vector<G4MaterialPropertyVector*> qq ; 
    S4MaterialPropertyVector::Import_VV(qq, fold ); 

    std::cout 
        << "vv.size " << vv.size() << "\n" 
        << "qq.size " << qq.size() << "\n" 
        ; 

    return 0 ; 
}

