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

   void Summary(const char* msg);

private:
   size_t m_length ;
   T* m_values ;
   T* m_domain ;
};



template <typename T>
GProperty<T>::GProperty( T* values, T* domain, size_t length)
{
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

