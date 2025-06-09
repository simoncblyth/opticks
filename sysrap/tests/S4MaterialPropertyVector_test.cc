/**
S4MaterialPropertyVector_test.cc
=================================

::

   ~/o/sysrap/tests/S4MaterialPropertyVector_test.sh



**/
#include <vector>
#include <iostream>

#include "ssys.h"
#include "NPX.h"
#include "S4MaterialPropertyVector.h"


struct S4MaterialPropertyVector_test
{
    static constexpr const  char* CETHETA = R"LITERAL(
array([[0.      , 0.911   ],
       [0.226893, 0.911   ],
       [0.488692, 0.9222  ],
       [0.715585, 0.9294  ],
       [0.959931, 0.9235  ],
       [1.151917, 0.93    ],
       [1.37881 , 0.9095  ],
       [1.48353 , 0.6261  ],
       [1.570796, 0.2733  ]])
)LITERAL" ;


    static void Populate( std::vector<G4MaterialPropertyVector*>& vv, int ni );
    static int VV();
    static int VV_CombinedArray();
    static int ConvertToArray();

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


int S4MaterialPropertyVector_test::ConvertToArray()
{
    std::cout << "[ConvertToArray\n" ;

    NP* cetheta = NPX::FromNumpyString<double>(CETHETA);
    G4MaterialPropertyVector* prop = S4MaterialPropertyVector::FromArray( cetheta );
    NP* cetheta_1 = S4MaterialPropertyVector::ConvertToArray(prop);
    bool same = NP::SameData(cetheta, cetheta_1);

    std::cout
        << "-ConvertToArray"
        << " cetheta " << ( cetheta ? cetheta->sstr() : "-" )
        << " cetheta_1 " << ( cetheta_1 ? cetheta_1->sstr() : "-" )
        << " same " << ( same ? "YES" : "NO " )
        << "\n"
        ;

    assert( same );

    bool reverse = true ;
    NP* cecosth = NP::MakeWithCosineDomain(cetheta, reverse);

    NPFold* fold = new NPFold ;
    fold->add("cetheta", cetheta );
    fold->add("cecosth", cecosth );
    fold->save("$FOLD/ConvertToArray");

    std::cout << "]ConvertToArray\n" ;
    return 0 ;
}






int S4MaterialPropertyVector_test::Main()
{
    const char* test = "ALL" ;
    const char* TEST = ssys::getenvvar("TEST", test);
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL || strcmp(TEST, "VV") == 0 )               rc += VV();
    if(ALL || strcmp(TEST, "VV_CombinedArray") == 0 ) rc += VV_CombinedArray();
    if(ALL || strcmp(TEST, "ConvertToArray") == 0 )   rc += ConvertToArray();

    std::cout << "S4MaterialPropertyVector_test::Main TEST [" << ( TEST ? TEST : "-" ) << "] rc " << rc << "\n" ;

    return rc ;
}

int main()
{
    return S4MaterialPropertyVector_test::Main() ;
}

