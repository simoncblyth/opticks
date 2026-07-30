#pragma once

/**
sblackbody.h
===============



* from the former npy/NPlanck.cpp

::

    In [1]: from scipy.constants import h,c,k
    In [2]: h,c,k
    Out[2]: (6.62607015e-34, 299792458.0, 1.380649e-23)

* https://physics.info/planck/
* http://www.fourmilab.ch/documents/specrend/specrend.c
* https://en.wikipedia.org/wiki/Planck%27s_law

TODO RESCUE::

   npy/ciexyz.h


**/

#include <cmath>
#include "srngcpu.h"
#include "NPFold.h"

template<typename T>
struct sblackbody
{
    int ni ;
    int nu ;

    T  temp_K ;
    T  nm0 ;
    T  nm1 ;

    int hd_factor ;

    NP* psr ;
    NP* cdf ;
    NP* icdf ;
    NP* icdf_prop ;
    NP* wavelength ;

    static double planck_spectral_radiance(double nm, double K=6500.) ;
    static double get_domain( int i, int ni, double nm0, double nm1);

    static NP* make_planck_spectral_radiance_one(int ni, T temp_K = 6500., T nm0 = 80., T nm1 = 800. );
    static NP* make_planck_spectral_radiance_set(int nj,                   T nm0 = 80., T nm1 = 800.);


    sblackbody(int ni, int nu, T temp_K, T nm0, T nm1, size_t num_wl);
    NP* generate_sample(size_t num_wl = 1'000'000 ) const ;
    T icdf_wavelength(T u) const ;


    NPFold* serialize() const ;
    void    save(const char* fold) const ;

};

/**
sblackbody::planck_spectral_radiance
-------------------------------------

Calculations always in double, storage uses templated type

**/


template<typename T>
inline double sblackbody<T>::planck_spectral_radiance(double nm, double K)   // static
{
    double h = 6.62606957e-34 ;
    double c = 299792458.0 ;
    double k = 1.3806488e-23 ;

    double a = 2.0*h*c*c ;
    double b = h*c/k ;

    double wlm = nm * 1e-9;

    return (a * pow(wlm, -5.0)) /
           (exp(b / (wlm * K)) - 1.0);
}


template<typename T>
inline double sblackbody<T>::get_domain( int i, int ni, double nm0, double nm1) // static
{
    return nm0 + (nm1-nm0)*double(i)/double(ni-1) ;
}


template<typename T>
inline NP* sblackbody<T>::make_planck_spectral_radiance_one(int ni, T _temp_K, T nm0, T nm1 ) // static
{
    int nj = 2 ;
    double temp_K = _temp_K ;
    NP* a = NP::Make<T>(ni, nj);
    T* aa = a->values<T>();
    for(int i=0 ; i < ni ; i += 1 )
    {
        double nm = get_domain(i, ni, nm0, nm1 );
        double psr = planck_spectral_radiance( nm, temp_K ) ;

        aa[i*nj + 0] = nm ;
        aa[i*nj + 1] = psr ;
    }
    return a ;
}

template<typename T>
inline NP* sblackbody<T>::make_planck_spectral_radiance_set(int nj, T nm0, T nm1)
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
            T nm = get_domain(j, nj, nm0, nm1 );
            T psr = planck_spectral_radiance( nm, temp_k ) ;
            int idx = i*nj*nk + j*nk ;
            aa[idx + 0] = nm ;
            aa[idx + 1] = psr ;
        }
    }
    return a ;
}



template<typename T>
inline sblackbody<T>::sblackbody(int _ni, int _nu, T _temp_K, T _nm0, T _nm1, size_t num_wl )
    :
    ni(_ni),
    nu(_nu),
    temp_K(_temp_K),
    nm0(_nm0),
    nm1(_nm1),
    hd_factor(0),
    psr(make_planck_spectral_radiance_one(ni,temp_K,nm0,nm1)),
    cdf(NP::MakeCDF<T>(psr)),
    icdf(NP::MakeICDF<T>(cdf, nu, hd_factor, false)),
    icdf_prop(nullptr),
    wavelength(nullptr)
{
    std::vector<NP::INT> shape0 = icdf->shape ;
    assert(shape0.size() == 3 && shape0[1] == nu );

    std::vector<NP::INT> shape1 = {nu} ;
    icdf->reshape(shape1);
    icdf_prop = NP::MakeProperty<double>(icdf, hd_factor );

    icdf->reshape(shape0); // restore original shape

    wavelength = generate_sample(num_wl);
}

template<typename T>
inline NP* sblackbody<T>::generate_sample(size_t num_wl ) const
{
    srngcpu rng ;
    NP* wavelength = rng.sample_icdf<T>( icdf_prop, num_wl );
    return wavelength ;
}


/**
sblackbody::icdf_wavelength
--------------------------------

Note that the interpolation must be done using the same type as icdf_prop

**/

template<typename T>
T sblackbody<T>::icdf_wavelength(T u) const
{
    T w = icdf_prop->interp<T>(u);
    return w ;
}



template<typename T>
inline NPFold* sblackbody<T>::serialize() const
{
    NPFold* f = new NPFold ;
    f->add("psr", psr );
    f->add("cdf", cdf );
    f->add("icdf", icdf );
    f->add("icdf_prop", icdf_prop );
    f->add("wavelength", wavelength );
    return f ;
}

template<typename T>
inline void sblackbody<T>::save(const char* fold) const
{
    NPFold* f = serialize();
    f->save(fold);
}


