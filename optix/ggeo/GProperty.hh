#ifndef GPROPERTY_H
#define GPROPERTY_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "assert.h"
#include "md5digest.hh"



template <class T>
class GDomain {
  public: 
     GDomain(T low, T high, T step) : m_low(low), m_high(high), m_step(step) {}
     virtual ~GDomain() {}
  public: 
     T getLow(){  return m_low ; }   
     T getHigh(){ return m_high ; }   
     T getStep(){ return m_step ; }
  public: 
     size_t getLength();   
     T* getValues();   

  private:
     T m_low ; 
     T m_high ; 
     T m_step ; 
};


template <typename T>
size_t GDomain<T>::getLength()
{
   T x = m_low ; 

   unsigned int n = 0 ;
   while( x <= m_high )
   {
      x += m_step ;
      n++ ; 
   }
   assert(n < 500); // sanity check 

   return n+1 ;
} 


template <typename T>
T* GDomain<T>::getValues()
{
   unsigned int length = getLength(); 
   T* domain = new T[length];
   for(unsigned int i=0 ; i < length ; i++)
   {
      domain[i] = m_low + i*m_step ; 
   }
   return domain ;
}




template <class T>
class GAry {
public: 
   GAry(unsigned int length, T* values=NULL) : m_length(length)
   {
       assert(length < 1000); //sanity check

       m_values = new T[m_length];
       if(values)
       {
           while(length--) m_values[length] = values[length] ;
       }
   } 
   virtual ~GAry() 
   {
       delete m_values ;
   }

   T getLeft(){  return m_values[0] ; }
   T getRight(){ return m_values[m_length-1] ; }
   T getValue(unsigned int index){ return m_values[index] ;}
   T* getValues(){ return m_values ; }
   size_t getLength(){ return m_length ; }
   size_t getNbytes(){ return m_length*sizeof(T) ; }

   int binary_search(T key)
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
private:
    T* m_values ; 
    unsigned int m_length ; 
};


template <typename T>
GAry<T>* np_interp(GAry<T>* xi, GAry<T>* xp, GAry<T>* fp )
{
    //
    // Loosely follow np.interp signature and implementation from 
    //    https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c#L599
    //

    assert(xp->getLength() == fp->getLength());  // input domain and values must be of same length

    // input domain and values
    T* dx = xp->getValues();   
    T* dy = fp->getValues();   
    T left = fp->getLeft();
    T right = fp->getRight();

    GAry<T>* res = new GAry<T>(xi->length); // Ary to be filled with interpolated values
    T* dres = res->getValues();

    for (unsigned int i = 0; i < res->length ; i++) 
    {
        const T z = xi->getValue(i);
        int j = xp->binary_search(z);

        if(j == -1)
        {
            dres[i] = left;
        }
        else if(j == xp->length - 1)
        {
            dres[i] = dy[j];
        }
        else if(j == xp->length )
        {
            dres[i] = right;
        }
        else
        {
            const T slope  = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
            dres[i] = slope*(z - dx[j]) + dy[j];
        }
    }
    return res ;
}




template <class T>
class GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;
   char* digest();   

   GProperty(T* values, T* domain, unsigned int length ) : m_length(length)
   {
       assert(length < 1000);
       m_values = new GAry<T>(length, values);
       m_domain = new GAry<T>(length, domain);
   }

   GProperty( GAry<T>* vals, GAry<T>* dom )  : m_values(vals), m_domain(dom) 
   {
       assert(vals->getLength() == dom->getLength());
       m_length = vals->getLength();
   }

   virtual ~GProperty()
   {
       delete m_values ;
       delete m_domain ;
   } 

public:
   GProperty<T> createInterpolatedProperty();
   GProperty<T> createInterpolatedProperty(GDomain<T>* domain);

public:
   void Summary(const char* msg);

private:
   unsigned int m_length ;
   GAry<T>* m_values ;
   GAry<T>* m_domain ;

};



template <typename T>
const char* GProperty<T>::DOMAIN_FMT = " %10.3f" ; 

template <typename T>
const char* GProperty<T>::VALUE_FMT = " %10.3f" ; 



template <typename T>
void GProperty<T>::Summary(const char* msg)
{
   //printf("%s : \n[%s]\n", msg, digest() );
   printf("%s : \n", msg );

   for(unsigned int i=0 ; i < m_length ; i++ )
   {
      if( i < 5 || i > m_length - 5 )
      {
          printf("%4u", i );
          printf(DOMAIN_FMT, m_domain->getValue(i));
          printf(VALUE_FMT,  m_values->getValue(i));
          printf("\n");
      }
   }
}


template <typename T>
char* GProperty<T>::digest()
{
    size_t v_nbytes = m_values->getNbytes();
    size_t d_nbytes = m_domain->getNbytes();
    assert(v_nbytes == d_nbytes);

    MD5Digest dig ;
    dig.update( (char*)m_values->getValues(), v_nbytes);
    dig.update( (char*)m_domain->getValues(), d_nbytes );
    return dig.finalize();
}

template <typename T>
GProperty<T> GProperty<T>::createInterpolatedProperty()
{
    // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
    GDomain<T> domain = new GDomain<T>(60.f, 810.f, 20.f ); 
    return createInterpolatedProperty(domain);
}


template <typename T>
GProperty<T> GProperty<T>::createInterpolatedProperty(GDomain<T>* domain)
{
    GAry<T>* idom = new GAry<T>(domain->getLength(), domain->getValues());
    GAry<T>* ival = np_interp( idom , m_domain, m_values );

    GProperty<T>* prop = new GProperty<T>( ival, idom );
    return prop ;
}



typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


#endif

