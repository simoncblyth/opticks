#pragma once

#include <vector>
#include <string>


template <typename T> class GDomain ; 
template <typename T> class GAry ; 
template <typename T> class GProperty ; 


#include "GGEO_API_EXPORT.hh"

template <class T>
class GGEO_API GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;
   static const T DELTA ; 
public: 
   static GProperty<T>* make_GROUPVEL(GProperty<T>* rindex);
   static GAry<T>*      make_dispersion_term(GProperty<T>* rindexE);
public: 
   static T maxdiff(GProperty<T>* a, GProperty<T>* b, bool dump=false);
   static bool hasSameDomain(GProperty<T>* a, GProperty<T>* b, T delta=-1, bool dump=false);
   static GProperty<T>* load(const char* path);
   static GProperty<T>* from_constant(T value, T* domain, unsigned int length );
   static GProperty<T>* from_constant(T value, T dlow, T dhigh);
   static GProperty<T>* make_one_minus(GProperty<T>* a);
   static GProperty<T>* make_addition(GProperty<T>* a, GProperty<T>* b, GProperty<T>* c=NULL, GProperty<T>* d=NULL );
   static std::string   make_table(int fwid, T dscale, bool dreciprocal,
                                   GProperty<T>* a, const char* atitle, 
                                   GProperty<T>* b, const char* btitle,
                                   GProperty<T>* c=NULL, const char* ctitle=NULL, 
                                   GProperty<T>* d=NULL, const char* dtitle=NULL,
                                   GProperty<T>* e=NULL, const char* etitle=NULL,
                                   GProperty<T>* f=NULL, const char* ftitle=NULL,
                                   GProperty<T>* g=NULL, const char* gtitle=NULL,
                                   GProperty<T>* h=NULL, const char* htitle=NULL
                                   );


   static std::string make_table(int fwid, T dscale, bool dreciprocal,  bool constant,
         std::vector< GProperty<T>* >& columns,
         std::vector< std::string >& titles 
         );

   static GProperty<T>* ramp(T low, T step, T* domain, unsigned int length );
   static GProperty<T>* planck_spectral_radiance(GDomain<T>* nm, T blackbody_temp_kelvin=6500.);
public:
   GProperty(GProperty<T>* other);
   GProperty(T* values, T* domain, unsigned int length );
   GProperty( GAry<T>* vals, GAry<T>* dom ); // stealing ctor, use with newly allocated GAry<T> 
   virtual ~GProperty();

public:
   void save(const char* path);
   void save(const char* dir, const char* reldir, const char* name);
   T getValue(unsigned int index);
   T getDomain(unsigned int index);
   T getInterpolatedValue(T val);
   unsigned int getLength();
   GAry<T>* getValues();
   GAry<T>* getDomain();
   void setValues(T val);
   char* digest();   
   std::string getDigestString();
public:
   bool isZero();
   bool isConstant();
   T getConstant(); 
   T getMin(); 
   T getMax(); 
public:
   // **lookup** here means that the input values are already within the domain 
   // this is appropriate for InverseCDF where the domain is 0:1 and 
   // the input values are uniform randoms within 0:1  
   //
   GAry<T>*      lookupCDF(GAry<T>* uvals);
   GAry<T>*      lookupCDF(unsigned int n);

   GAry<T>*      lookupCDF_ValueLookup(GAry<T>* uvals);
   GAry<T>*      lookupCDF_ValueLookup(unsigned int n);



   // **sample** means that must do binary values search to locate relevant index
   // this is appropriate for CDF where domain is arbitrary and values 
   // range from 0:1 the input being uniform randoms within 0:1
   //  
   GAry<T>*      sampleCDF(GAry<T>* uvals);
   GAry<T>*      sampleCDF(unsigned int n);

   GAry<T>*      sampleCDFDev(unsigned int n);
   GProperty<T>* createCDFTrivially();
   GProperty<T>* createCDF();
   GProperty<T>* createReversedReciprocalDomain(T scale=1);
   GProperty<T>* createSliced(int ifr, int ito);
   GProperty<T>* createZeroTrimmed();  // trims extremes to create GProperty with at most one zero value entry at either end  
   GProperty<T>* createInverseCDF(unsigned int n=0);
   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);

public:
   void SummaryV(const char* msg, unsigned int nline=5);
   void Summary(const char* msg="GProperty::Summary", unsigned int imod=5 );
   std::string brief(const char* msg=""); 

private:
   unsigned int m_length ;
   GAry<T>*     m_values ;
   GAry<T>*     m_domain ;

};




typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


