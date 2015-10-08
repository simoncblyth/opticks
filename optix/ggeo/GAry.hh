#pragma once

template <class T>
class GAry {

public: 
   static GAry<T>* urandom(unsigned int n=100); 
   static GAry<T>* create_from_floats(unsigned int length, float* values);
   static GAry<T>* product(GAry<T>* a, GAry<T>* b);
   static GAry<T>* subtract(GAry<T>* a, GAry<T>* b);
   static GAry<T>* add(GAry<T>* a, GAry<T>* b);
   static T        maxdiff(GAry<T>* a, GAry<T>* b);
   static GAry<T>* from_constant(unsigned int length, T value );
   static GAry<T>* zeros(unsigned int length);
   static GAry<T>* ones(unsigned int length);
   static GAry<T>* ramp(unsigned int length, T low, T step );
   static GAry<T>* linspace(T num, T start=0, T stop=1);
   static T            step(T num, T start=0, T stop=1);
   static GAry<T>* np_interp(GAry<T>* xi, GAry<T>* xp, GAry<T>* fp );
   static T np_interp(const T z, GAry<T>* xp, GAry<T>* fp );

public: 
   GAry<T>* copy();
   GAry(GAry<T>* other);
   GAry(unsigned int length, T* values=0);
   virtual ~GAry();

public: 
   GAry<T>* cumsum(unsigned int offzero=0);
   GAry<T>* diff(); // domain bin widths
   GAry<T>* mid();  // average of values at bin edges, ie linear approximation of mid bin value 
   GAry<T>* reversed(bool reciprocal=false);
   GAry<T>* sliced(int ifr, int ito);
   void save(const char* path);

public: 
   T getLeft(){                              return m_values[0] ; }
   T getRight(){                             return m_values[m_length-1] ; }
   T getValue(unsigned int index){           return m_values[index] ;}
   T* getValues(){                           return m_values ; }
   unsigned int getLength(){                 return m_length ; }
   unsigned int getNbytes(){                 return m_length*sizeof(T) ; }

public: 
   T min(unsigned int& idx); 
   T max(unsigned int& idx); 
   T getValueFractional(T findex); // fractional bin
   T getValueLookup(T u);          // from u(0:1) to fractional bin to values
   unsigned int getLeftZero();
   unsigned int getRightZero();

public: 
   void setValue(unsigned int index, T val){ m_values[index] = val ;}

public: 
   void Summary(const char* msg="GAry::Summary", unsigned int imod=1, T presentation_scale=1.0);
   void scale(T sc);
   void add(GAry<T>* other);
   void subtract(GAry<T>* other);
   void reciprocate();

   // find the index of the value closest to the random draw u on the low side
   int binary_search(T u);
   int linear_search(T u);
   T fractional_binary_search(T u);  // like binary search but provides the fractional bin too

   unsigned int sample_cdf(T u);

private:
    T* m_values ; 
    unsigned int m_length ; 
};




typedef GAry<float> GAryF ;
typedef GAry<double> GAryD ;


