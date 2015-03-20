#ifndef GPROPERTY_H
#define GPROPERTY_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "assert.h"
#include "md5digest.hh"

#include "GDomain.hh"
#include "GAry.hh"


template <class T>
class GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;
   char* digest();   


   static GProperty<T>* from_constant(T value, T* domain, unsigned int length ) 
   {
       GAry<T>* vals = GAry<T>::from_constant(length, value );
       GAry<T>* doms = new GAry<T>(length, domain);
       return new GProperty<T>( vals, doms );
   }

   static GProperty<T>* ramp(T low, T step, T* domain, unsigned int length ) 
   {
       GAry<T>* vals = GAry<T>::ramp(length, low, step );
       GAry<T>* doms = new GAry<T>(length, domain);
       return new GProperty<T>( vals, doms );
   }


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

   T getValue(unsigned int index)
   {
       return m_values->getValue(index);
   }
   T getDomain(unsigned int index)
   {
       return m_domain->getValue(index);
   }
   T getInterpolatedValue(T val);
 



public:
   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);

public:
   void Summary(const char* msg, unsigned int nline=5);

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
void GProperty<T>::Summary(const char* msg, unsigned int nline )
{
   if(nline == 0) return ;
   printf("%s : \n", msg );
   for(unsigned int i=0 ; i < m_length ; i++ )
   {
      if( i < nline || i > m_length - nline )
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
GProperty<T>* GProperty<T>::createInterpolatedProperty(GDomain<T>* domain)
{
    GAry<T>* idom = new GAry<T>(domain->getLength(), domain->getValues());
    GAry<T>* ival = np_interp( idom , m_domain, m_values );

    GProperty<T>* prop = new GProperty<T>( ival, idom );
    return prop ;
}

template <typename T>
T GProperty<T>::getInterpolatedValue(T val)
{
    return np_interp( val , m_domain, m_values );
}


typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


#endif

