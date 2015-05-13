#ifndef GPROPERTY_H
#define GPROPERTY_H

#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <math.h> 
#include "assert.h"
#include "md5digest.hh"

#include "GDomain.hh"
#include "GAry.hh"
#include "NPY.hpp"


template <class T>
class GProperty {
public: 
   static const char* DOMAIN_FMT ;
   static const char* VALUE_FMT ;
   char* digest();   
   std::string getDigestString();


   static GProperty<T>* load(const char* path)
   {
       NPY* npy = NPY::load(path);
       npy->Summary();

       assert(npy->getDimensions() == 2);

       unsigned int ni = npy->getShape(0);
       unsigned int nj = npy->getShape(1);
       assert(nj == 2);
       T* data = npy->getFloats();

       GAry<T>* doms = new GAry<T>(ni);
       GAry<T>* vals = new GAry<T>(ni);

       for(unsigned int i=0 ; i < ni ; i++){
       for(unsigned int j=0 ; j < nj ; j++)
       {
            unsigned int index = i*nj + j ;
            T v = data[index];
            switch(j)
            {
               case 0:doms->setValue(i, v); break ; 
               case 1:vals->setValue(i, v); break ;
            }      
            //printf(" i %u j %u index %u  val %10.3f \n", i,j, index, v ); 
       }
       }
       return new GProperty<T>(vals, doms) ; 
   }


   static T maxdiff(GProperty<T>* a, GProperty<T>* b)
   {
       assert(a->getLength() == b->getLength());
       T mv(0);
       T md(0);
       for(unsigned int i=0 ; i < a->getLength() ; i++)
       {
           T av = a->getValues()->getValue(i) ;
           T bv = b->getValues()->getValue(i) ;
           T dv = fabs(av-bv); 

           if(dv > mv) mv = dv ;
           //printf("av %10.3f bv %10.3f dv*1e9 %10.3f mv*1e9 %10.3f \n", av, bv, dv*1e9, mv*1e9); 

           T ad = a->getDomain()->getValue(i) ;
           T bd = b->getDomain()->getValue(i) ;
           T dd = fabs(ad-bd); 
           if(dd > md) md = dd ;
           //printf("ad %10.3f bd %10.3f dd*1e9 %10.3f md*1e9 %10.3f \n", ad, bd, dd*1e9, md*1e9); 
       }

       return mv > md ? mv : md  ; 
   }  

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

   GProperty(GProperty<T>* other) : m_length(other->getLength())
   {
       m_values = new GAry<T>(other->getValues());
       m_domain = new GAry<T>(other->getDomain());
   }

   GProperty(T* values, T* domain, unsigned int length ) : m_length(length)
   {
       assert(length < 1000);
       m_values = new GAry<T>(length, values);
       m_domain = new GAry<T>(length, domain);
   }

   // stealing ctor, use with newly allocated GAry<T> 
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

   unsigned int getLength()
   {
       assert(m_values->getLength() == m_domain->getLength());
       return m_domain->getLength(); 
   }
 

public:
   GAry<T>* getValues()
   {
       return m_values ; 
   }
   GAry<T>* getDomain()
   {
       return m_domain ; 
   }

   GProperty<T>* createCDF();
   GProperty<T>* createReciprocalCDF();

public:
   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);

public:
   void SummaryV(const char* msg, unsigned int nline=5);
   void Summary(const char* msg, unsigned int imod=5 );

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
void GProperty<T>::SummaryV(const char* msg, unsigned int nline )
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
void GProperty<T>::Summary(const char* msg, unsigned int imod )
{
   char* pdig = digest();
   printf("%s : %s : %u \n", msg, pdig, m_length );
   free(pdig);

   for(unsigned int p=0 ; p < 2 ; p++)
   {
       switch(p)
       {
           case 0:printf("d ");break;
           case 1:printf("v ");break;
       }
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
GProperty<T>* GProperty<T>::createCDF()
{
    GAry<T>* x = new GAry<T>(getDomain());  // copy 
    GAry<T>* cy = getValues()->cumsum();     
    cy->scale(1./cy->getRight());            // normalise by making RHS 1.
    return new GProperty<T>( cy, x );        // stealing ctor 
}

/*
101 def construct_cdf( xy ):
102     """
103     :param xy:
104 
105     Creating cumulative density functions needed by chroma, 
106     eg for generating a wavelengths of reemitted photons.::
107 
...
146     assert len(xy.shape) == 2 and xy.shape[-1] == 2
147     x,y  = xy[:,0], xy[:,1]
148     cy = np.cumsum(y)
149     cdf_y = cy/cy[-1]   # normalize to 1 at RHS
150     return np.vstack([x,cdf_y]).T
*/



template <typename T>
GProperty<T>* GProperty<T>::createReciprocalCDF()
{
    bool reciprocal = true ; 
    GAry<T>* x = getDomain()->reversed(reciprocal);   // 1/nm in reverse order 
    GAry<T>* y = getValues()->reversed();

    // ymid, xdif, prod have one bin less as need pairs 
    GAry<T>* ymid = y->mid() ;  
    GAry<T>* xdif = x->diff() ;  
    GAry<T>* prod = GAry<T>::product( ymid, xdif );

    unsigned int offzero = 1 ;               // gives one extra zero bin
    GAry<T>* cy = prod->cumsum(offzero);
    cy->scale(1./cy->getRight());            // normalise by making RHS 1.

    bool debug = false ; 
    if(debug)
    {
        unsigned int imod = x->getLength()/20  ; 
        T psc = 1000.f ; // presentation scale
        x->Summary("x [domain reversed reciprocal]", imod, psc);
        y->Summary("y [values reversed]", imod, psc);
        ymid->Summary("ymid: y->mid()", imod, psc);
        xdif->Summary("xdif: x->diff()", imod, psc);
        prod->Summary("prod:  ymid*xdif ", imod, psc);
        cy->Summary("cy: prod->cumsum(1) scaled  ", imod, psc);
    }

    delete y ;
    delete ymid ;
    delete xdif ;
    delete prod ; 

    return new GProperty<T>( cy, x );        // stealing ctor 
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

