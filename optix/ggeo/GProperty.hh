#ifndef GPROPERTY_H
#define GPROPERTY_H

#include <string>
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
   std::string getDigestString();


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
   static GProperty<T>* createCDF(GProperty<T>* intensity);
   static GProperty<T>* createReciprocalCDF(GProperty<T>* intensity);

public:
   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);

public:
   void Summary(const char* msg, unsigned int nline=5);
   void SummaryH(const char* msg, unsigned int imod=5 );

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
void GProperty<T>::SummaryH(const char* msg, unsigned int imod )
{
   char* pdig = digest();
   printf("%s : %s : %u \n", msg, pdig, m_length );
   free(pdig);

   for(unsigned int p=0 ; p < 2 ; p++)
   {
       for(unsigned int i=0 ; i < m_length ; i++ )
       {
           if( i % imod == 0 )
           { 
               switch(p)
               {
                   case 0:printf(DOMAIN_FMT, m_domain->getValue(i));break;
                   case 1:printf(VALUE_FMT,  m_values->getValue(i));break;
               }
           }
       }
       printf("\n");
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
std::string GProperty<T>::getDigestString()
{
    return digest();
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

template <typename T>
GProperty<T>* GProperty<T>::createCDF(GProperty<T>* dist)
{
    return dist ; 
}

template <typename T>
GProperty<T>* GProperty<T>::createReciprocalCDF(GProperty<T>* dist)
{
    return dist ; 
}

//  translation of NumPy based env/geant4/geometry/collada/collada_to_chroma.py::construct_cdf_energywise
/*
154 def construct_cdf_energywise(xy):
155     """
156     Duplicates DsChromaG4Scintillation::BuildThePhysicsTable     
157     """
158     assert len(xy.shape) == 2 and xy.shape[-1] == 2
159 
160     bcdf = np.empty( xy.shape )
161 
162     rxy = xy[::-1]              # reverse order, for ascending energy 
163 
164     x = 1/rxy[:,0]              # work in inverse wavelength 1/nm
165 
166     y = rxy[:,1]
167 
168     ymid = (y[:-1]+y[1:])/2     # looses entry as needs pair
169 
170     xdif = np.diff(x)
171 
172     #bcdf[:,0] = rxy[:,0]        # back to wavelength
173     bcdf[:,0] = x                # keeping 1/wavelenth
174 
175     bcdf[0,1] = 0.
176 
177     np.cumsum(ymid*xdif, out=bcdf[1:,1])
178 
179     bcdf[1:,1] = bcdf[1:,1]/bcdf[1:,1].max()
180 
181     return bcdf
*/





typedef GProperty<float>  GPropertyF ;
typedef GProperty<double> GPropertyD ;


#endif

