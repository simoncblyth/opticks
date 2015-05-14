#include "GAry.hh"

#include "assert.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>


template <typename T>
T GAry<T>::np_interp(const T z, GAry<T>* xp, GAry<T>* fp )
{
    // input domain and values
    T* dx = xp->getValues();   
    T* dy = fp->getValues();   
    T left = fp->getLeft();
    T right = fp->getRight();

    T ival ;
    int j = xp->binary_search(z);
    if(j == -1)
    {
        ival = left;
    }
    else if(j == xp->getLength() - 1)
    {
        ival = dy[j];
    }
    else if(j == xp->getLength() )
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
    GAry<T>* result = new GAry<T>(a->getLength()); // Ary to be filled with interpolated values
    for (unsigned int i = 0; i < result->getLength() ; i++) 
    {
       T ab = b->getValue(i) - a->getValue(i);
       result->setValue(i, ab );
    }
    return result ;
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
GAry<T>* np_reversed(GAry<T>* y, bool reciprocal)
{
    unsigned int len = y->getLength();
    GAry<T>* ry = new GAry<T>(len); 

    T one(1); 
    for (unsigned int i = 0; i < len ; i++)
    {
        T val = y->getValue(i) ; 
        ry->setValue(len-1-i, reciprocal ? one/val : val );
    }
    return ry ;
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
GAry<T>* GAry<T>::create_from_floats(unsigned int length, float* values)
{
    // promoting floats to doubles
    T* tvalues = new T[length];
    for(unsigned int i=0 ; i < length ; i++) tvalues[i] = values[i];
    GAry<T>* ary = new GAry<T>(length, tvalues);
    delete tvalues ;       
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
GAry<T>* GAry<T>::from_constant(unsigned int length, T value )
{
    GAry<T>* ary = new GAry<T>( length, NULL );
    T* vals = ary->getValues();
    for(unsigned int i=0 ; i < length; i++) vals[i] = value ;
    return ary ;
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
GAry<T>::GAry(GAry<T>* other) : m_length(other->getLength())
{
    m_values = new T[m_length];
    for(unsigned int i=0 ; i < m_length; i++) m_values[i] = other->getValue(i) ;
}

template <typename T>
GAry<T>::GAry(unsigned int length, T* values) : m_length(length)
{
    assert(length < 1000); //sanity check
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
GAry<T>* GAry<T>::cumsum(unsigned int offzero)
{  
    return np_cumsum(this, offzero) ; 
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
GAry<T>* GAry<T>::reversed(bool reciprocal)
{ 
     return np_reversed(this, reciprocal) ; 
}


template <typename T>
void GAry<T>::Summary(const char* msg, unsigned int imod, T presentation_scale)
{
    printf("%s length %u \n", msg, m_length);
    for(unsigned int i=0 ; i < m_length ; i++ ) if(i%imod == 0) printf(" %10.3f ", getValue(i)*presentation_scale);
    printf("\n");
} 


template <typename T>
void GAry<T>::scale(T sc)
{
    for(unsigned int i=0 ; i < m_length ; i++ ) m_values[i] *= sc ; 
}  

template <typename T>
int GAry<T>::binary_search(T key)
{
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
    return imin - 1; 
} 




/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class GAry<float>;
template class GAry<double>;



