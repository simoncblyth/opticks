
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <string>

#include "GDomain.hh"
#include "Opticks.hh"

#include "PLOG.hh"


template <typename T>
GDomain<T>* GDomain<T>::fDefaultDomain = NULL  ;

template <typename T>
GDomain<T>* GDomain<T>::GetDefaultDomain()  // static
{
    if(fDefaultDomain == NULL)
    {
        fDefaultDomain = new GDomain<T>(Opticks::DOMAIN_LOW, Opticks::DOMAIN_HIGH, Opticks::DOMAIN_STEP ); 
    }
    return fDefaultDomain ;
}


template <typename T>
unsigned GDomain<T>::Length(T low, T high, T step) // static
{
   T x = low ; 

   unsigned int n = 0 ;
   while( x <= high )
   {
      x += step ;
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
GDomain<T>* GDomain<T>::makeInterpolationDomain(T step) const 
{
   return new GDomain<T>(m_low, m_high, step);
}


template <typename T>
GDomain<T>::GDomain(T low, T high, T step) 
    : 
    m_low(low), 
    m_high(high), 
    m_step(step),
    m_length(Length(m_low, m_high, m_step))
{
}



template <typename T>
std::string GDomain<T>::desc() const
{
    std::stringstream ss ; 
    ss
       << " GDomain " 
       << " low " << m_low
       << " high " << m_high
       << " step " << m_step
       << " length " << m_length
    ;
    return ss.str();
}


template <typename T>
void GDomain<T>::Summary(const char* msg) const 
{
   LOG(info) << msg 
             << " low " << m_low
             << " high " << m_high
             << " step " << m_step
             << " length " << m_length
             ;

   T* vals = getValues();
   std::stringstream ss ;  
   for(unsigned int i=0 ; i < m_length ; i++) ss << i << ":" << vals[i] << " " ; 

   LOG(info) << "values: " << ss.str() ;
}



template <typename T>
bool GDomain<T>::isEqual(GDomain<T>* other) const 
{
    return 
       getLow() == other->getLow()   &&
       getHigh() == other->getHigh() &&
       getStep() == other->getStep() ;
}


template <typename T>
T* GDomain<T>::getValues() const 
{
   T* domain = new T[m_length];
   for(unsigned int i=0 ; i < m_length ; i++)
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


