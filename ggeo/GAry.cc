   
#include <cstddef>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "NPY.hpp"
#include "NPlanck.hpp"
#include "NCIE.hpp"

#include "GAry.hh"
#include "GDomain.hh"

#include "PLOG.hh"



template <typename T>
T GAry<T>::np_interp(const T z, GAry<T>* xp, GAry<T>* fp )
{
    // input domain and values
    //   
    //  :param z:   "x" domain value for which the interpolated "y" value is to be obtained
    //  :param xp : domain x-coordinates of the data points, must be increasing.
    //  :param fp : value  y-coordinates of the data points, same length as `xp`.
    //  
    //  :return:   y value (ie linear interpolation of fp values) corresponding to the ordinate supplied 
    //   

    T* dx = xp->getValues();   

    T* dy = fp->getValues();   
    T left = fp->getLeft();
    T right = fp->getRight();

    int len = xp->getLength();
 
    assert( len > 0 );
 
    if(dy[len-1] != right )
    {
        LOG(warning) << "GAry<T>::np_interp "
                     << " len " << len
                     << " dy[len-1] " << dy[len-1]
                     << " right " << right
                     << " left " << left
                     ;
    }

    assert(dy[len-1] == right );


/*
   This assert is firing occasionally... but unreproducibly 

[2015-Jul-22 15:28:16.828602]: GProperty::save 2d array of length 275 to : /tmp/reemissionCDF.npy
createZeroTrimmed ifr 0 ito 273 
np_sliced ifr 0 ito 273  alen 275 blen 273 
np_sliced ifr 0 ito 273  alen 275 blen 273 
GBoundaryLib::createReemissionBuffer icdf  : 570b234e132f398d4213400cc88f427b : 4096 
d       0.000      0.063      0.125      0.188      0.250      0.313      0.375      0.438      0.500      0.563      0.625      0.688      0.750      0.813      0.875      0.938
v     799.898    463.793    450.234    441.314    434.113    428.762    424.561    420.824    417.127    413.177    408.946    404.907    401.409    398.164    394.649    389.214
[2015-Jul-22 15:28:16.834756]: GProperty::save 2d array of length 4096 to : /tmp/invertedReemissionCDF.npy
Assertion failed: (dy[len-1] == right), function np_interp, file /Users/blyth/env/optix/ggeo/GAry.cc, line 39.
Abort trap: 6

[2015-Jul-24 10:30:36.403821]: GProperty::save 2d array of length 4096 to : /tmp/invertedReemissionCDF.npy
[2015-Jul-24 10:30:36.443812]: GAry<T>::np_interp  len 4096 dy[len-1] nan right nan left 799.898
Assertion failed: (dy[len-1] == right), function np_interp, file /Users/blyth/env/optix/ggeo/GAry.cc, line 50.
/Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 550:  2369 Abort trap: 6           $bin $*

*/

    T ival ;
    int j = xp->binary_search(z);  // find low side domain index corresponding to domain value z
    if(j == -1)
    {
        ival = left;
    }
    else if(j == len - 1)
    {
        ival = dy[j]; // right 
    }
    else if(j == len )
    {
        ival = right;
    }
    else
    {
        const T slope  = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
        ival = slope*(z - dx[j]) + dy[j];
    }
    return ival ; 
}


template <typename T>
GAry<T>* GAry<T>::np_interp(GAry<T>* xi, GAry<T>* xp, GAry<T>* fp )
{
    //
    // Loosely follow np.interp signature and implementation from 
    //    https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c#L599
    //

    assert(xp->getLength() == fp->getLength());  // input domain and values must be of same length

    GAry<T>* res = new GAry<T>(xi->getLength()); // Ary to be filled with interpolated values
    T* dres = res->getValues();

    for (unsigned int i = 0; i < res->getLength() ; i++) 
    {
        dres[i] = np_interp( xi->getValue(i), xp, fp ) ;
    }
    return res ;
}

template <typename T>
GAry<T>* np_product(GAry<T>* a, GAry<T>* b)
{
    assert(a->getLength() == b->getLength()); 
    GAry<T>* prod = new GAry<T>(a->getLength()); // Ary to be filled with interpolated values
    for (unsigned int i = 0; i < prod->getLength() ; i++) 
    {
       T ab = a->getValue(i) * b->getValue(i);
       prod->setValue(i, ab );
    }
    return prod ;
}

template <typename T>
GAry<T>* np_subtract(GAry<T>* a, GAry<T>* b)
{
    assert(a->getLength() == b->getLength()); 
    GAry<T>* result = new GAry<T>(a->getLength()); 
    for (unsigned int i = 0; i < result->getLength() ; i++) 
    {
       T ab = b->getValue(i) - a->getValue(i);
       result->setValue(i, ab );
    }
    return result ;
}


template <typename T>
GAry<T>* np_add(GAry<T>* a, GAry<T>* b)
{
    assert(a->getLength() == b->getLength()); 
    GAry<T>* result = new GAry<T>(a->getLength()); 
    for (unsigned int i = 0; i < result->getLength() ; i++) 
    {
       T ab = b->getValue(i) + a->getValue(i);
       result->setValue(i, ab );
    }
    return result ;
}


template <typename T>
GAry<T>* np_clip(GAry<T>* a, GAry<T>* low, GAry<T>* high, GAry<T>* low_fallback, GAry<T>* high_fallback )
{
    assert(a->getLength() == low->getLength()); 
    assert(a->getLength() == high->getLength()); 
    if(low_fallback) assert(a->getLength() == low_fallback->getLength()); 
    if(high_fallback) assert(a->getLength() == high_fallback->getLength()); 

    GAry<T>* cc = new GAry<T>(a->getLength()); 
    for (unsigned int i = 0; i < cc->getLength() ; i++) 
    {
        T aa = a->getValue(i);
        T lo = low->getValue(i);
        T hi = high->getValue(i);

        if( aa < lo )
        {
            cc->setValue(i, low_fallback ? low_fallback->getValue(i) : lo );
        }
        else if(aa > hi )
        {
            cc->setValue(i, high_fallback ? high_fallback->getValue(i) : hi );
        }
        else
        {
            cc->setValue(i, aa );

        }
    }
    return cc ;
}





template <typename T>
T np_maxdiff(GAry<T>* a, GAry<T>* b, bool dump)
{

    unsigned alen = a->getLength() ;
    unsigned blen = b->getLength() ;
 
    if(dump) LOG(info) << " np_maxdiff " 
                       << " a " << std::setw(5) << alen
                       << " b " << std::setw(5) << blen
                       ;

    assert(alen == blen); 
    T max(0);
    for (unsigned int i = 0; i < a->getLength() ; i++) 
    {
       T av = a->getValue(i);
       T bv = b->getValue(i);
       T ab = bv - av ;

       max = std::max( max, ab );

       if(dump) LOG(info) 
                    <<  " i " << std::setw(4) << i
                    <<  " av " << std::setw(10) << av 
                    <<  " bv " << std::setw(10) << bv 
                    <<  " ab " << std::setw(10) << ab 
                    ; 

    }

    if(dump) LOG(info) << " maxdiff " << max ; 

    return max ;
}






template <typename T>
GAry<T>* np_cumsum(GAry<T>* y, unsigned int offzero)
{
    unsigned int len = y->getLength();
    GAry<T>* cy = new GAry<T>(len+offzero); // Ary to be filled with cumsum values
    T sum(0);
    for (unsigned int j = 0; j < offzero ; j++) 
    {
        cy->setValue(j, 0);
    }
    for (unsigned int i = 0; i < len ; i++) 
    {
        sum += y->getValue(i);
        cy->setValue(offzero+i, sum);
    }
    return cy ;
}

template <typename T>
GAry<T>* np_reversed(GAry<T>* y, bool reciprocal, T scale)
{
    unsigned int len = y->getLength();
    GAry<T>* ry = new GAry<T>(len); 

    for (unsigned int i = 0; i < len ; i++)
    {
        T val = y->getValue(i) ; 
        ry->setValue(len-1-i, reciprocal ? scale/val : scale*val );
    }
    return ry ;
}


template <typename T>
GAry<T>* GAry<T>::reciprocal(GAry<T>* y, T scale)
{
    unsigned int len = y->getLength();
    GAry<T>* ry = new GAry<T>(len); 
    for (unsigned int i = 0; i < len ; i++)
    {
        T val = y->getValue(i) ; 
        ry->setValue(i, scale/val );
    }
    return ry ;
}




template <typename T>
GAry<T>* g4_groupvel_bintrick(GAry<T>* y)
{
   /*
    This curious bin shifting is used by G4 GROUPVEL calc

    In order not to loose a bin
    a tricky manoever of using the 1st and last bin and 
    the average of the body bins
    which means the first bin is half width, and last is 1.5 width
    
    ::

            0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
            |    /     /     /     /      |
            |   /     /     /     /       |
            0  1     2     3     4        5        <--- still 6 


    bb = np.zeros_like(aa)
    bb[0] = aa[0]
    bb[1:-1] = (aa[1:-1] + aa[:-2])/2.
    bb[-1] = aa[-1]
    return bb

   */

    GAry<T>& Y = *y ;
    unsigned int len = y->getLength();
    GAry<T>* ry = new GAry<T>(len); 

    ry->setValue(0    , Y[0]);
    ry->setValue(len-1, Y[-1]);

    T two(2);

    for (unsigned i = 1; i < len - 1 ; i++)
    {
        T val = (Y[i-1] + Y[i])/two ; 
        ry->setValue(i, val );
    }
    return ry ;
}

template <typename T>
GAry<T>* np_gradient(GAry<T>* y)
{
    /*
    Take a look at np.gradient??

    The gradient is computed using second order accurate central differences
    in the interior and either first differences or second order accurate 
    one-sides (forward or backwards) differences at the boundaries. The
    returned gradient hence has the same shape as the input array.

          0    <--- (0,1)  
  
          1    <--- (0,2)/2
          2    <--- (1,3)/2
          3    <--  (2,4)/2
          4    <--  (3,5)/2
          5    <--  (4,6)/2
          6    <--  (5,7)/2
          7    <--  (6,8)/2
          8    <--  (7,9)/2

          9    <--  (8,9) 

    ::

        out = np.zeros_like(y)
        out[0] = y[1] - y[0]
        out[1:-1] = (y[2:]-y[:-2])/2.0
        out[-1] = y[-1] - y[-2]
        return o
    */

    GAry<T>& Y = *y ;
    unsigned int len = y->getLength();
    assert(len >= 2 );
    T two(2);

    GAry<T>* g = new GAry<T>(len); 
    g->setValue(0    , Y[1] - Y[0] );
    for (unsigned i = 1; i < len - 1 ; i++) g->setValue(i, (Y[i+1] - Y[i-1])/two );
    g->setValue(len-1, Y[-1] - Y[-2]);

    return g ;
}


template <typename T>
GAry<T>* np_sliced(GAry<T>* a, int ifr, int ito)
{
   /*

In [13]: a = np.linspace(0,1,11)

In [14]: a
Out[14]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])

In [16]: a[0:-1]
Out[16]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])

In [17]: a[0:11]
Out[17]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])

   */
    int alen = a->getLength();
    if(ifr < 0 ) ifr += alen ;   
    if(ito < 0 ) ito += alen ;   
    assert(ifr >=0 && ifr <  alen);
    assert(ito >=0 && ito <= alen);

    int blen = ito - ifr ;   // py style 0-based one-beyond "ito"
    assert( ito >= ifr );

    //printf("Gary.cc:np_sliced ifr %d ito %d  alen %u blen %u \n", ifr, ito, alen, blen );  

    GAry<T>* b = new GAry<T>(blen); 

    for (int ia = 0; ia < alen ; ia++)
    {
        if( ia >= ifr && ia < ito )
        {
            unsigned int ib = ia - ifr ;
            T val = a->getValue(ia) ; 
            b->setValue( ib, val );
        }
    }     
    return b ;
}





template <typename T>
GAry<T>* np_diff(GAry<T>* y)
{
    unsigned int len = y->getLength() ;
    GAry<T>* dy = new GAry<T>(len - 1); 
    for (unsigned int i = 0; i < len - 1 ; i++)
    {
        T diff = y->getValue(i+1) - y->getValue(i) ; 
        dy->setValue(i, diff );
    }
    return dy ;
}

template <typename T>
GAry<T>* np_mid(GAry<T>* y)
{
    unsigned int len = y->getLength() ;
    GAry<T>* my = new GAry<T>(len - 1); 
    T two(2);
    for (unsigned int i = 0; i < len - 1 ; i++)
    {
        T mid = (y->getValue(i) + y->getValue(i+1))/two ; 
        my->setValue(i, mid );
    }
    return my ;
}


template <typename T>
GAry<T>* GAry<T>::urandom(unsigned int length)
{
    GAry<T>* u = new GAry<T>(length); 
    T* vals = u->getValues();

    typedef boost::mt19937          RNG_t;
    typedef boost::uniform_real<>   Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    RNG_t rng;
    Distrib_t distrib(0,1);
    Generator_t gen(rng, distrib);    

    for (unsigned int i = 0; i < length ; i++) vals[i] = gen();
    return u ;
}


template <typename T>
GAry<T>* GAry<T>::from_domain(GDomain<T>* domain)
{
    GAry<T>* ary = new GAry<T>( domain->getLength(), domain->getValues() );
    return ary ; 
}


template <typename T>
GAry<T>* GAry<T>::planck_spectral_radiance(GAry<T>* nm, T blackbody_temp_kelvin)
{
    T* nmv = nm->getValues();
    GAry<T>* ary = new GAry<T>( nm->getLength(), NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < ary->getLength(); i++) vals[i] = ::planck_spectral_radiance(nmv[i], blackbody_temp_kelvin) ;  
    return ary ;
}

template <typename T>
GAry<T>* GAry<T>::cie_weight(GAry<T>* nm, unsigned int component)
{
    T* nmv = nm->getValues();
    GAry<T>* ary = new GAry<T>( nm->getLength(), NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < ary->getLength(); i++) 
    {
        switch(component)
        {
            case 0:vals[i] = NCIE::X(nmv[i]) ;break;  
            case 1:vals[i] = NCIE::Y(nmv[i]) ;break;  
            case 2:vals[i] = NCIE::Z(nmv[i]) ;break;  
        }
    }
    return ary ;
}


template <typename T>
GAry<T>* GAry<T>::cie_X(GAry<T>* nm)
{
    return cie_weight(nm, 0);
}
template <typename T>
GAry<T>* GAry<T>::cie_Y(GAry<T>* nm)
{
    return cie_weight(nm, 1);
}
template <typename T>
GAry<T>* GAry<T>::cie_Z(GAry<T>* nm)
{
    return cie_weight(nm, 2);
}









template <typename T>
GAry<T>* GAry<T>::create_from_floats(unsigned int length, float* values)
{
    // promoting floats to doubles
    T* tvalues = new T[length];
    for(unsigned int i=0 ; i < length ; i++) tvalues[i] = values[i];
    GAry<T>* ary = new GAry<T>(length, tvalues);
    delete[] tvalues ;       
    return ary ;
}

template <typename T>
GAry<T>* GAry<T>::product(GAry<T>* a, GAry<T>* b)
{
    return np_product(a, b);
}

template <typename T>
GAry<T>* GAry<T>::subtract(GAry<T>* a, GAry<T>* b)
{
    return np_subtract(a, b);
}
template <typename T>
GAry<T>* GAry<T>::add(GAry<T>* a, GAry<T>* b)
{
    return np_add(a, b);
}


template <typename T>
GAry<T>* GAry<T>::clip(GAry<T>* low, GAry<T>* high, GAry<T>* low_fallback, GAry<T>* high_fallback )
{
    return np_clip(this, low, high, low_fallback, high_fallback );
}


template <typename T>
T GAry<T>::maxdiff(GAry<T>* a, GAry<T>* b, bool dump)
{
    return np_maxdiff(a, b, dump);
}

template <typename T>
GAry<T>* GAry<T>::reciprocal(T scale_)
{
    return reciprocal(this, scale_);
}






template <typename T>
GAry<T>* GAry<T>::from_constant(unsigned int length, T value )
{
    GAry<T>* ary = new GAry<T>( length, NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < length; i++) vals[i] = value ;
    return ary ;
} 


template <typename T>
GAry<T>* GAry<T>::zeros(unsigned int length)
{
    return GAry<T>::from_constant(length, 0);
} 

template <typename T>
GAry<T>* GAry<T>::ones(unsigned int length)
{
    return GAry<T>::from_constant(length, 1);
} 





template <typename T>
GAry<T>* GAry<T>::ramp(unsigned int length, T low, T step )
{
    GAry<T>* ary = new GAry<T>( length, NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < length; i++) vals[i] = low + step*i ;  
    return ary ;
} 


template <typename T>
T GAry<T>::step(T num, T start, T stop)
{
    return (stop - start)/(num - 1) ; 
}

template <typename T>
GAry<T>* GAry<T>::linspace(T num, T start, T stop)
{
   /*
In [11]: np.linspace(0,1,10)
Out[11]: 
array([ 0.   ,  0.111,  0.222,  0.333,  0.444,  0.556,  0.667,  0.778,
        0.889,  1.   ])

In [12]: np.linspace(0,1,11)
Out[12]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])

   */
    T step_ = step(num, start, stop);
    GAry<T>* ary = new GAry<T>( num, NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < num ; i++) vals[i] = start + step_*i ;  
    return ary ;
} 


template <typename T>
GAry<T>* GAry<T>::copy() 
{
    return new GAry<T>(this);
}

template <typename T>
GAry<T>::GAry(GAry<T>* other) : m_length(other->getLength())
{
    m_values = new T[m_length];
    for(unsigned int i=0 ; i < m_length; i++) m_values[i] = other->getValue(i) ;
}

template <typename T>
GAry<T>::GAry(unsigned int length, T* values) : m_length(length)
{
    m_values = new T[m_length];
    if(values)
    {
        for(unsigned int i=0 ; i < length; i++) m_values[i] = values[i] ;
    }
} 

template <typename T>
GAry<T>::~GAry() 
{
    delete m_values ;
}



template <typename T>
void GAry<T>::setValues(T val)
{
    for(unsigned int i=0 ; i < m_length; i++) m_values[i] = val ;
}


template <typename T>
GAry<T>* GAry<T>::cumsum(unsigned int offzero)
{  
    return np_cumsum(this, offzero) ; 
}

template <typename T>
GAry<T>* GAry<T>::g4_groupvel_bintrick()
{  
    return ::g4_groupvel_bintrick(this) ; 
}

template <typename T>
GAry<T>* GAry<T>::gradient()
{  
    return ::np_gradient(this) ; 
}







template <typename T>
GAry<T>* GAry<T>::diff()
{ 
    return np_diff(this) ; 
}     // domain bin widths

template <typename T>
GAry<T>* GAry<T>::mid()
{ 
    return np_mid(this) ; 
}  // average of values at bin edges, ie linear approximation of mid bin "y" value 

template <typename T>
GAry<T>* GAry<T>::reversed(bool reciprocal_, T scale_)
{ 
     return np_reversed(this, reciprocal_, scale_) ; 
}

template <typename T>
GAry<T>* GAry<T>::sliced(int ifr_, int ito_)
{ 
     return np_sliced(this, ifr_, ito_) ; 
}



template <typename T>
void GAry<T>::Summary(const char* msg, unsigned int imod, T presentation_scale)
{
    printf("%s length %u  leftzero %u rightzero %u \n", msg, m_length, getLeftZero(), getRightZero() );

    if(m_length < 100)
    {
        for(unsigned int i=0 ; i < m_length ; i++ ) if(i%imod == 0) printf(" %10.3f ", getValue(i)*presentation_scale);
        printf("\n");
    }
    else
    {
        for(unsigned int i=0 ; i < m_length ; i++ ) 
        {
            if( i < 10 || i > m_length - 10 )
            {
                printf(" %10.3f ", getValue(i)*presentation_scale);
            } 
            else if ( i == 10 )
            {
                printf(" ... ");
            }
        }
        printf("\n");
   }

    T rng[2] ;
    unsigned int idx[2] ;
    rng[0] = min(idx[0]);
    rng[1] = max(idx[1]);

    printf("mi/mx idx %u %u  range %15.5f %15.5f  1/range %15.5f %15.5f \n", idx[0], idx[1], rng[0], rng[1], 1./rng[0], 1./rng[1] );

} 


template <typename T>
T GAry<T>::getValueFractional(T findex)
{
    unsigned int idx(findex);
    T dva ; 

    if(idx + 1 < m_length )
    {
       T frac(findex - T(idx));
       dva = m_values[idx]*(1.-frac) + m_values[idx+1]*frac ;
    }
    else
    {
       dva = m_values[m_length-1]; 
    } 
    return dva ; 
}

template <typename T>
T GAry<T>::getValueLookup(T u)
{
    // convert u (0:1) into fractional bin, and use that to lookup values
    assert( u <= 1. && u >= 0. );
    T findex = u * (m_length - 1) ; 
    return getValueFractional(findex);
}

template <typename T>
unsigned int GAry<T>::getLeftZero()
{
    // when the values start with a string of zeros, 
    // return the index of rightmost such zero minus 1, 
    // otherwise return 0

    T zero(0);
    int ifr(0);
    for(unsigned int i=0 ; i < m_length ; i++)
    {
        if( m_values[i] == zero ) ifr = i  ;
        else
            break ; 
    }

    return ifr > 0 ? ifr - 1 : 0 ; 
}


template <typename T>
unsigned int GAry<T>::getRightZero()
{
    // when the values end with a string of zeros, 
    // return the lowest index + 1, 
    // otherwise return m_length

    T zero(0);
    unsigned int ito(m_length);
    for(unsigned int i=0 ; i < m_length ; i++)
    {
        unsigned int j = m_length - 1 - i ;   // looks at bins from the right 
        if( m_values[j] == zero ) ito = j  ;
        else
            break ; 
    }
    return ito < m_length ? ito + 1 : m_length ; 
}



template <typename T>
void GAry<T>::add(GAry<T>* other)
{  
    assert(other->getLength() == m_length);
    for(unsigned int i=0 ; i < m_length ; i++ ) m_values[i] += other->getValue(i) ; 
}

template <typename T>
void GAry<T>::subtract(GAry<T>* other)
{  
    assert(other->getLength() == m_length);
    for(unsigned int i=0 ; i < m_length ; i++ ) m_values[i] -= other->getValue(i) ; 
}




template <typename T>
void GAry<T>::scale(T sc)
{
    for(unsigned int i=0 ; i < m_length ; i++ ) m_values[i] *= sc ; 
}  

template <typename T>
void GAry<T>::reciprocate()
{
    T one(1);
    for(unsigned int i=0 ; i < m_length ; i++ ) m_values[i] = one/m_values[i] ; 
}  

template <typename T>
int GAry<T>::linear_search(T key)
{
    // for checking edge case behaviour of binary_search
    // expected to return same values as binary_search more slowly

    if(key < m_values[0])          
    {
        return -1 ;        // indicates "below-lower-bound"
    }
    else if(key > m_values[m_length-1]) 
    {
        return m_length ;   // indicates "above-upper-bound"
    }
    else if(key == m_values[m_length-1])
    {
        return m_length - 1 ;   //  at upper bound   
    }
    else
    {
        for(unsigned int i=0 ; i < m_length - 1 ; ++i )
        {
            //   m_values[0] : m_values[1]
            //   ...
            //   m_values[m_length-2] : m_values[m_length-1]   
            //
            if(key >= m_values[i] && key < m_values[i+1]) return i ;
        }
    }  
    assert(0); // not expected here
    return -2 ; 
}
 


template <typename T>
int GAry<T>::binary_search(T key)
{
   // Find bin index containing the key   
   //
   // :param key: value to be "placed" within the sequence of ascending values
   // :return: index of the low side value  
   //
   //      * normally index is range 0:m_length-1
   //      * if the key exceeds m_values[m_length-1] m_length is returned, meaning "to the right" 
   //
   // NB this is used by np_interp 

    if(key > m_values[m_length-1])
    {
        return m_length ;
    }
    unsigned int imin = 0 ; 
    unsigned int imax = m_length ; 
    unsigned int imid ;   

    while(imin < imax)
    {
        imid = imin + ((imax - imin) >> 1);
        if (key >= m_values[imid]) 
        {
            imin = imid + 1;
        }
        else 
        {
            imax = imid;
        }
    }

    assert( imin == imax );
    //if(imin != imax ) printf("GAry<T>::binary_search key %10.5f  imin %u imax %u len %u \n", key, imin, imax, m_length );

    return imin - 1; 
} 


template <typename T>
T GAry<T>::fractional_binary_search(T u)
{
    // the advantage in dealing in fractional indices
    // is can delay resorting to using domain information for a bit longer

    int idx = binary_search(u);
    T frac  = (u - m_values[idx])/(m_values[idx+1]-m_values[idx]);  // fraction of bin 
    return T(idx) + frac ; 
}



template <typename T>
unsigned int GAry<T>::sample_cdf(T u)
{
    // other than edge cases, this gives same results as binary_search

    int lower = 0;
    int upper = m_length - 1;

    while(lower < upper-1)
    {   
        int half = (lower + upper) / 2;
        if (u < m_values[half])
        {
            upper = half;
        }
        else 
        {
            lower = half;
        }
    }   

    assert( lower == upper - 1 );
    return lower ; 
}


template <typename T>
T GAry<T>::min(unsigned int& idx)
{
   T mi(FLT_MAX);
   for(unsigned int i=0 ; i < m_length ; i++ )
   {
       T v = m_values[i];
       if(v < mi)
       {
           mi = v ; 
           idx = i ;
       }
   }
   return mi ; 
}

template <typename T>
T GAry<T>::max(unsigned int& idx)
{
   T mx(-FLT_MAX);
   for(unsigned int i=0 ; i < m_length ; i++ )
   {
       T v = m_values[i];
       if(v > mx)
       {
           mx = v ; 
           idx = i ;
       }
   }
   return mx ; 
}


template <typename T>
void GAry<T>::save(const char* path)
{
    std::vector<int> shape ; 
    shape.push_back(m_length);

    std::string metadata = "{}" ; 

    std::vector<T> data ; 
    for(unsigned int i=0 ; i < m_length ; i++ ) data.push_back(m_values[i]) ;

    LOG(info) << "GAry::save 1d array of length " << m_length << " to : " << path ;  
    NPY<T> npy(shape, data, metadata);
    npy.save(path);
}




/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class GAry<float>;
template class GAry<double>;   // needs work on NPY for this



