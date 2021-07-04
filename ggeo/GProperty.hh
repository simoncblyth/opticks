/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <vector>
#include <string>


template <typename T> class NPY ; 
template <typename T> class GDomain ; 
template <typename T> class GAry ; 
template <typename T> class GProperty ; 


#include "GGEO_API_EXPORT.hh"

/**
GProperty
==========

* values on a domain


**/

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
   static void copy_values(GProperty<T>* to, GProperty<T>* fr, T domdelta=1e-4);
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
   GProperty<T>* copy() const ;
   GProperty(const GProperty<T>* other);
   GProperty(T* values, T* domain, unsigned int length );
   GProperty( GAry<T>* vals, GAry<T>* dom ); // stealing ctor, use with newly allocated GAry<T> 
   virtual ~GProperty();

public:
   void save(const char* path);
   void save(const char* dir, const char* name);
   void save(const char* dir, const char* reldir, const char* name);
   T getValue(unsigned index) const ;
   T getDomain(unsigned index) const ;
   T getInterpolatedValue(T val) const ;
   unsigned getLength() const ;
   GAry<T>* getValues() const ;
   GAry<T>* getDomain() const ;

   NPY<T>*  makeArray() const ; 

   char* digest();   
   std::string getDigestString();
public:
   void copyValuesFrom(GProperty<T>* other, T domdelta=1e-4 );
   void setValues(T val);
public:
   bool isZero() const ;
   bool isConstant() const ;
   T getConstant() const ; 
   T getMin() const ; 
   T getMax() const ; 
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
   std::string brief(const char* msg="") const ; 

private:
   unsigned int m_length ;
   GAry<T>*     m_values ;
   GAry<T>*     m_domain ;

};




typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


