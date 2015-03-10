#ifndef GPROPERTY_H
#define GPROPERTY_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "md5digest.hh"

template <class T>
class GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;

   char* digest();   

   GProperty(T* values, T* domain, size_t len );
   virtual ~GProperty();

public:
   void setStandardDomain(T low, T high, T step);
   T getLow();   
   T getHigh();   
   T getStep();   
   T* getInterpolatedValues();
   T* getInterpolatedDomain();

public:
   void Summary(const char* msg);

private:
   size_t m_length ;
   T* m_values ;
   T* m_domain ;

private:
   T m_low ; 
   T m_high ; 
   T m_step ; 

};









template <typename T>
GProperty<T>::GProperty( T* values, T* domain, size_t length)
{
    // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)

    size_t nbytes = length*sizeof(T);
    m_length = length ; 

    m_values = (T*)malloc( nbytes );
    m_domain = (T*)malloc( nbytes );

    memcpy( m_values, values, nbytes ); 
    memcpy( m_domain, domain, nbytes ); 
}

template <typename T>
GProperty<T>::~GProperty()
{
    free( m_values );
    free( m_domain );
}


template <typename T>
const char* GProperty<T>::DOMAIN_FMT = " %10.3f" ; 

template <typename T>
const char* GProperty<T>::VALUE_FMT = " %10.3f" ; 


template <typename T>
void GProperty<T>::setStandardDomain( T low, T high, T step)
{
   m_low  = low ;
   m_high = high ;
   m_step = step ;
}

template <typename T>
T GProperty<T>::getLow()
{
   return m_low ; 
}

template <typename T>
T GProperty<T>::getHigh()
{
   return m_high ; 
}

template <typename T>
T GProperty<T>::getStep()
{
   return m_step ; 
}

template <typename T>
T* GProperty<T>::getInterpolatedValues()
{

   // chroma/chroma/gpu/geometry.py
   //
   //  67         def interp_material_property(domain, prop):
   //  68             """
   //  69             :param domain: usually wavelengths for reemission_cdf 1/wavelengths[::-1]
   //  70             :param prop:
   //  71 
   //  72             note that it is essential that the material properties be
   //  73             interpolated linearly. this fact is used in the propagation
   //  74             code to guarantee that probabilities still sum to one.
   //  75             """
   //  76             ascending_ = lambda _:np.all(np.diff(_) >= 0)
   //  77             assert ascending_(domain)
   //  78             assert ascending_(prop[:,0])
   //  79             return np.interp(domain, prop[:,0], prop[:,1]).astype(np.float32)
   //
   // https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/function_base.py#L1117
   // https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c
   // https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c#L599
   //
   // TODO: reimplement what arr_interp is doing 
   //


   return NULL ; 
}

template <typename T>
T* GProperty<T>::getInterpolatedDomain()
{
   return NULL ; 
}




 




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
          printf(DOMAIN_FMT, m_domain[i]);
          printf(VALUE_FMT,  m_values[i]);
          printf("\n");
      }
   }
}




template <typename T>
char* GProperty<T>::digest()
{
    size_t nbytes = m_length*sizeof(T);

    MD5Digest dig ;
    dig.update( (char*)m_values, nbytes);
    dig.update( (char*)m_domain, nbytes );
    return dig.finalize();

}





typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


#endif

