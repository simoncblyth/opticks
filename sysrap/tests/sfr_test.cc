/**

~/o/sysrap/tests/sfr_test.sh
~/o/sysrap/tests/sfr_test.cc
~/o/sysrap/tests/sfr_test.py

**/

#include "ssys.h"
#include "sfr.h"

struct sfr_test
{
    static int main();
    static int MakeFromAxis();
    static int hello();
};

int sfr_test::main()
{
    //const char* test = "ALL" ;
    const char* test = "MakeFromAxis" ;
    const char* TEST = ssys::getenvvar("TEST", test);
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"hello")) rc += hello();
    if(ALL||0==strcmp(TEST,"MakeFromAxis")) rc += MakeFromAxis();
    return rc ;
}

int sfr_test::MakeFromAxis()
{
    const char* tpde = "45,45,0,1000" ;
    sfr mf = sfr::MakeFromAxis<double>(tpde, ',');

    std::cout
         << "sfr_test::MakeFromAxis"
         << " tpde " << tpde
         << "\n"
         << mf.desc()
         << "\n"
         ;
    return 0 ;
}

int sfr_test::hello()
{
    sfr a ;
    a.set_name("hello a");

    a.aux0.x = 1 ;
    a.aux0.y = 2 ;
    a.aux0.z = 3 ;
    a.aux0.w = 4 ;

    a.aux1.x = 10 ;
    a.aux1.y = 20 ;
    a.aux1.z = 30 ;
    a.aux1.w = 40 ;

    a.aux2.x = 100 ;
    a.aux2.y = 200 ;
    a.aux2.z = 300 ;
    a.aux2.w = 400 ;

    std::cout << "A\n" << a.desc() << std::endl;
    a.save("$FOLD");

    sfr b = sfr::Load("$FOLD");
    std::cout << "B\n" << b.desc() << std::endl;

    return 0;
}

int main()
{
    return sfr_test::main();
}
