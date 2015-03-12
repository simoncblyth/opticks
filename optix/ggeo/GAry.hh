#ifndef GARY_H
#define GARY_H

#include "assert.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>


template <class T>
class GAry {
public: 
   static GAry<T>* create_from_floats(unsigned int length, float* values)
   {
       // promoting floats to doubles
       T* tvalues = new T[length];
       for(unsigned int i=0 ; i < length ; i++) tvalues[i] = values[i];
       GAry<T>* ary = new GAry<T>(length, tvalues);
       delete tvalues ;       
       return ary ;
   }

   GAry(unsigned int length, T* values=NULL) : m_length(length)
   {
       assert(length < 1000); //sanity check

       m_values = new T[m_length];
       if(values)
       {
           for(unsigned int i=0 ; i < length; i++) m_values[i] = values[i] ;
       }
   } 

   static GAry<T>* from_constant(unsigned int length, T value )
   {
       GAry<T>* ary = new GAry<T>( length, NULL );
       double* vals = ary->getValues();
       for(unsigned int i=0 ; i < length; i++) vals[i] = value ;
       return ary ;
   } 

   virtual ~GAry() 
   {
       delete m_values ;
   }

   T getLeft(){  return m_values[0] ; }
   T getRight(){ return m_values[m_length-1] ; }
   T getValue(unsigned int index){ return m_values[index] ;}
   T* getValues(){ return m_values ; }
   size_t getLength(){ return m_length ; }
   size_t getNbytes(){ return m_length*sizeof(T) ; }

   int binary_search(T key)
   {
        if(key > m_values[m_length-1])
        {
            return m_length ;
        }
        unsigned int imin = 0 ; 
        unsigned int imax = m_length ; 
        unsigned int imid ;   

        while(imin < imax)
        {
            imid = imin + ((imax - imin) >> 1);
            if (key >= m_values[imid]) 
            {
                imin = imid + 1;
            }
            else 
            {
                imax = imid;
            }
        }
        return imin - 1; 
    } 
private:
    T* m_values ; 
    unsigned int m_length ; 
};


template <typename T>
GAry<T>* np_interp(GAry<T>* xi, GAry<T>* xp, GAry<T>* fp )
{
    //
    // Loosely follow np.interp signature and implementation from 
    //    https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c#L599
    //

    assert(xp->getLength() == fp->getLength());  // input domain and values must be of same length

    // input domain and values
    T* dx = xp->getValues();   
    T* dy = fp->getValues();   
    T left = fp->getLeft();
    T right = fp->getRight();

    GAry<T>* res = new GAry<T>(xi->getLength()); // Ary to be filled with interpolated values
    T* dres = res->getValues();

    for (unsigned int i = 0; i < res->getLength() ; i++) 
    {
        const T z = xi->getValue(i);
        int j = xp->binary_search(z);

        if(j == -1)
        {
            dres[i] = left;
        }
        else if(j == xp->getLength() - 1)
        {
            dres[i] = dy[j];
        }
        else if(j == xp->getLength() )
        {
            dres[i] = right;
        }
        else
        {
            const T slope  = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
            dres[i] = slope*(z - dx[j]) + dy[j];
        }
    }
    return res ;
}


#endif
