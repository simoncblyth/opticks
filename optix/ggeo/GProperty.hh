#pragma once

#include "GAry.hh"
#include "GDomain.hh"
#include <string>


template <class T>
class GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;
public: 
   static T maxdiff(GProperty<T>* a, GProperty<T>* b);
   static GProperty<T>* load(const char* path);
   static GProperty<T>* from_constant(T value, T* domain, unsigned int length );
   static GProperty<T>* ramp(T low, T step, T* domain, unsigned int length );

public:
   GProperty(GProperty<T>* other);
   GProperty(T* values, T* domain, unsigned int length );
   GProperty( GAry<T>* vals, GAry<T>* dom ); // stealing ctor, use with newly allocated GAry<T> 
   virtual ~GProperty();

public:
   T getValue(unsigned int index);
   T getDomain(unsigned int index);
   T getInterpolatedValue(T val);
   unsigned int getLength();
   GAry<T>* getValues();
   GAry<T>* getDomain();
   char* digest();   
   std::string getDigestString();

public:
   GProperty<T>* createCDF();
   GProperty<T>* createReciprocalCDF();
   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);

public:
   void SummaryV(const char* msg, unsigned int nline=5);
   void Summary(const char* msg, unsigned int imod=5 );

private:
   unsigned int m_length ;
   GAry<T>*     m_values ;
   GAry<T>*     m_domain ;

};



template <typename T>
inline T GProperty<T>::getValue(unsigned int index)
{
    return m_values->getValue(index);
}

template <typename T>
inline T GProperty<T>::getDomain(unsigned int index)
{
    return m_domain->getValue(index);
}

 
template <typename T>
GAry<T>* GProperty<T>::getValues()
{
    return m_values ; 
}

template <typename T>
GAry<T>* GProperty<T>::getDomain()
{
    return m_domain ; 
}



typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;



