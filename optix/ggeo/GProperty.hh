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
   void save(const char* path);
   T getValue(unsigned int index);
   T getDomain(unsigned int index);
   T getInterpolatedValue(T val);
   unsigned int getLength();
   GAry<T>* getValues();
   GAry<T>* getDomain();
   char* digest();   
   std::string getDigestString();

public:
   // **lookup** here means that the input values are already within the domain 
   // this is appropriate for InverseCDF where the domain is 0:1 and 
   // the input values are uniform randoms within 0:1  
   //
   GAry<T>*      lookupCDF(GAry<T>* uvals);
   GAry<T>*      lookupCDF(unsigned int n);

   // **sample** means that must do binary values search to locate relevant index
   // this is appropriate for CDF where domain is arbitrary and values 
   // range from 0:1 the input being uniform randoms within 0:1
   //  
   GAry<T>*      sampleCDF(GAry<T>* uvals);
   GAry<T>*      sampleCDF(unsigned int n);

   GAry<T>*      sampleCDFDev(unsigned int n);
   GProperty<T>* createCDFTrivially();
   GProperty<T>* createCDF(bool reciprocal_domain=false);
   GProperty<T>* createInverseCDF(unsigned int n=0);
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



