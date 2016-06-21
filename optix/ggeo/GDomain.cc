
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <string>

#include "GDomain.hh"

#include "PLOG.hh"


template <typename T>
void GDomain<T>::Summary(const char* msg)
{
   unsigned int len = getLength();

   LOG(info) << msg 
             << " low " << m_low
             << " high " << m_high
             << " step " << m_step
             << " len " << len 
             ;

   T* vals = getValues();
   std::stringstream ss ;  
   for(unsigned int i=0 ; i < len ; i++) ss << i << ":" << vals[i] << " " ; 

   LOG(info) << "values: " << ss.str() ;
}



template <typename T>
bool GDomain<T>::isEqual(GDomain<T>* other)
{
    return 
       getLow() == other->getLow()   &&
       getHigh() == other->getHigh() &&
       getStep() == other->getStep() ;
}


template <typename T>
unsigned int GDomain<T>::getLength()
{
   T x = m_low ; 

   unsigned int n = 0 ;
   while( x <= m_high )
   {
      x += m_step ;
      n++ ; 
   }
   assert(n < 5000); // sanity check 

   //return n+1 ;  
   //   **old-off-by-one-bug** that has been hiding due to another bug in the domain, 
   //   was 810nm which did not fit in with the step of 20 and num of 39  
   //   the final n increment is not felt by <= m_high check, so the necessary extra
   //   +1 happened already inside the while 
   //
   return n ;
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

/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class GDomain<float>;
template class GDomain<double>;


