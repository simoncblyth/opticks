/**
S4MaterialPropertyVector_test.cc
=================================

::

   ~/o/sysrap/tests/S4MaterialPropertyVector_test.sh 



**/
#include <vector>
#include <iostream>

#include "ssys.h"
#include "S4MaterialPropertyVector.h"


struct S4MaterialPropertyVector_test
{
    static void Populate( std::vector<G4MaterialPropertyVector*>& vv, int ni ); 
    static int VV(); 
    static int VV_CombinedArray(); 
    static int Main(); 
};


void S4MaterialPropertyVector_test::Populate( std::vector<G4MaterialPropertyVector*>& vv, int ni )
{
    vv.resize(ni) ; 
    for( int i=0 ; i < ni ; i++) vv[i] = S4MaterialPropertyVector::Make_V(double(i*10.)) ; 
}


int S4MaterialPropertyVector_test::VV()
{
    std::vector<G4MaterialPropertyVector*> vv ; 
    Populate(vv,10); 

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

int S4MaterialPropertyVector_test::VV_CombinedArray()
{
    std::vector<G4MaterialPropertyVector*> vv ; 
    Populate(vv,10); 

    NP* vvcom = S4MaterialPropertyVector::Serialize_VV_CombinedArray(vv) ; 

    std::cout << "S4MaterialPropertyVector_test::VV_CombinedArray vvcom " << vvcom->desc() << "\n" ; 

    std::vector<G4MaterialPropertyVector*> qq ; 
    S4MaterialPropertyVector::Import_VV_CombinedArray(qq, vvcom ); 

    std::cout 
        << "vv.size " << vv.size() << "\n" 
        << "qq.size " << qq.size() << "\n" 
        ; 

    vvcom->save("$FOLD/VV_CombinedArray.npy"); 

    return 0 ; 
}


int S4MaterialPropertyVector_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 
    int rc = 0 ; 
    if(ALL || strcmp(TEST, "VV") == 0 )               rc += VV(); 
    if(ALL || strcmp(TEST, "VV_CombinedArray") == 0 ) rc += VV_CombinedArray(); 

    std::cout << "S4MaterialPropertyVector_test::Main TEST " << ( TEST ? TEST : "-" ) << " rc " << rc << "\n" ; 

    return rc ; 
} 

int main()
{
    return S4MaterialPropertyVector_test::Main() ; 
}

