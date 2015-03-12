#ifndef GDOMAIN_H
#define GDOMAIN_H

#include "assert.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>


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


#endif
