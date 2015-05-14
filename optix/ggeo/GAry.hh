#pragma once

template <class T>
class GAry {

public: 
   static GAry<T>* urandom(unsigned int n=100); 
   static GAry<T>* create_from_floats(unsigned int length, float* values);
   static GAry<T>* product(GAry<T>* a, GAry<T>* b);
   static GAry<T>* subtract(GAry<T>* a, GAry<T>* b);
   static GAry<T>* from_constant(unsigned int length, T value );
   static GAry<T>* ramp(unsigned int length, T low, T step );
   static GAry<T>* np_interp(GAry<T>* xi, GAry<T>* xp, GAry<T>* fp );
   static T np_interp(const T z, GAry<T>* xp, GAry<T>* fp );

public: 
   GAry(GAry<T>* other);
   GAry(unsigned int length, T* values=0);
   virtual ~GAry();

public: 
   GAry<T>* cumsum(unsigned int offzero=0);
   GAry<T>* diff(); // domain bin widths
   GAry<T>* mid();  // average of values at bin edges, ie linear approximation of mid bin value 
   GAry<T>* reversed(bool reciprocal=false);

public: 
   T getLeft(){                              return m_values[0] ; }
   T getRight(){                             return m_values[m_length-1] ; }
   T getValue(unsigned int index){           return m_values[index] ;}
   void setValue(unsigned int index, T val){ m_values[index] = val ;}
   T* getValues(){                           return m_values ; }
   unsigned int getLength(){                 return m_length ; }
   unsigned int getNbytes(){                 return m_length*sizeof(T) ; }

public: 
   void Summary(const char* msg, unsigned int imod=1, T presentation_scale=1.0);
   void scale(T sc);
   int binary_search(T key);

private:
    T* m_values ; 
    unsigned int m_length ; 
};




typedef GAry<float> GAryF ;
typedef GAry<double> GAryD ;


