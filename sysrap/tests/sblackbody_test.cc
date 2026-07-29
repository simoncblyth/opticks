#include "ssys.h"
#include "sblackbody.h"
#include "NPFold.h"

struct sblackbody_test
{
    static int planck_spectral_radiance_set();
    static int planck_sample();

    static int main();
};


inline int sblackbody_test::planck_spectral_radiance_set()
{
    int nj = 2000;
    NP* a = sblackbody<double>::make_planck_spectral_radiance_set(nj);
    a->save("$FOLD/planck_spectral_radiance_set.npy");
    return 0 ;
}

inline int sblackbody_test::planck_sample()
{
    sblackbody<double> bb(4096,4096,6500.,80.,800., 1'000'000 );
    bb.save("$FOLD");
    return 0 ;
}

inline int sblackbody_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","ALL");
    bool ALL = 0 == strcmp(TEST, "ALL") ;
    int rc = 0;
    if(ALL||0==strcmp(TEST,"planck_spectral_radiance_set")) rc += planck_spectral_radiance_set();
    if(ALL||0==strcmp(TEST,"planck_sample")) rc += planck_sample();
    return rc ;
}

int main()
{
    return sblackbody_test::main();
}
