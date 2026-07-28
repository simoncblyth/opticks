#include "ssys.h"
#include "sblackbody.h"
#include "srngcpu.h"
#include "NPFold.h"

struct sblackbody_test
{
    template<typename T>
    static NP* make_planck_spectral_radiance_set(int nj);

    template<typename T>
    static NP* make_planck_spectral_radiance_one(int ni, int temp_K);

    static int planck_spectral_radiance_set();
    static int planck_cdf();
    static int planck_icdf();
    static int planck_sample();

    static int main();
};



template<typename T>
inline NP* sblackbody_test::make_planck_spectral_radiance_set(int nj)
{
    //std::vector<int> temps = {{6500,6000,5500,5000,4500}} ;
    std::vector<int> temps = {{7000,6000,5000,4000,3000}} ;
    int ni = temps.size();
    int nk = 2 ;

    NP* a = NP::Make<T>(ni, nj, nk);
    double* aa = a->values<T>();
    for(int i=0 ; i < ni ; i += 1 )
    {
        int temp = temps[i] ;
        a->names.push_back(std::to_string(temp));
        T temp_k = temp ;

        for(int j=0 ; j < nj ; j += 1 )
        {
            T nm = T(j+100) ;
            T psr = sblackbody::planck_spectral_radiance( nm, temp_k ) ;
            int idx = i*nj*nk + j*nk ;
            aa[idx + 0] = nm ;
            aa[idx + 1] = psr ;
        }
    }
    return a ;
}

inline int sblackbody_test::planck_spectral_radiance_set()
{
    int nj = 2000;
    NP* a = make_planck_spectral_radiance_set<double>(nj);
    a->save("$FOLD/planck_spectral_radiance_set.npy");
    return 0 ;
}





template<typename T>
inline NP* sblackbody_test::make_planck_spectral_radiance_one(int ni, int _temp_K)
{
    T temp_k = _temp_K ;
    int nk = 2 ;

    NP* a = NP::Make<T>(ni, 2);
    double* aa = a->values<T>();
    for(int i=0 ; i < ni ; i += 1 )
    {
        T nm = T(i+100) ;
        T psr = sblackbody::planck_spectral_radiance( nm, temp_k ) ;
        aa[i*nk + 0] = nm ;
        aa[i*nk + 1] = psr ;
    }
    return a ;
}

inline int sblackbody_test::planck_cdf()
{
    int ni = 1000 ;
    NP* psr = make_planck_spectral_radiance_one<double>(ni, 6500);
    NP* cdf = NP::MakeCDF<double>(psr);

    NPFold* f = new NPFold ;
    f->add("psr", psr );
    f->add("cdf", cdf );

    f->save("$FOLD");
    return 0 ;
}

inline int sblackbody_test::planck_icdf()
{
    int ni = 1000 ;
    NP* psr = make_planck_spectral_radiance_one<double>(ni, 6500);
    NP* cdf = NP::MakeCDF<double>(psr);

    unsigned nu = 2000 ;
    unsigned hd_factor = 0 ;
    bool dump = false ;
    NP* icdf = NP::MakeICDF<double>( cdf, nu, hd_factor, dump );
    std::vector<NP::INT> shape = {nu} ;
    icdf->reshape(shape);

    NP* icdf_prop = NP::MakeProperty<double>(icdf, hd_factor );

    NPFold* f = new NPFold ;
    f->add("psr", psr );
    f->add("cdf", cdf );
    f->add("icdf", icdf );
    f->add("icdf_prop", icdf_prop );

    f->save("$FOLD");
    return 0 ;
}


inline int sblackbody_test::planck_sample()
{
    int ni = 1000 ;
    NP* psr = make_planck_spectral_radiance_one<double>(ni, 6500);
    NP* cdf = NP::MakeCDF<double>(psr);

    unsigned nu = 2000 ;
    unsigned hd_factor = 0 ;
    bool dump = false ;
    NP* icdf = NP::MakeICDF<double>( cdf, nu, hd_factor, dump );
    std::vector<NP::INT> shape = {nu} ;
    icdf->reshape(shape);

    NP* icdf_prop = NP::MakeProperty<double>(icdf, hd_factor );

    size_t num_wl = 1'000'000 ;
    srngcpu rng ;
    NP* wavelength = rng.sample_icdf<double>( icdf_prop, num_wl );


    NPFold* f = new NPFold ;
    f->add("psr", psr );
    f->add("cdf", cdf );
    f->add("icdf", icdf );
    f->add("icdf_prop", icdf_prop );
    f->add("wavelength", wavelength );

    f->save("$FOLD");
    return 0 ;
}


inline int sblackbody_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","ALL");
    bool ALL = 0 == strcmp(TEST, "ALL") ;
    int rc = 0;
    if(ALL||0==strcmp(TEST,"planck_spectral_radiance_set")) rc += planck_spectral_radiance_set();
    if(ALL||0==strcmp(TEST,"planck_cdf"))  rc += planck_cdf();
    if(ALL||0==strcmp(TEST,"planck_icdf")) rc += planck_icdf();
    if(ALL||0==strcmp(TEST,"planck_sample")) rc += planck_sample();
    return rc ;
}

int main()
{
    return sblackbody_test::main();
}
