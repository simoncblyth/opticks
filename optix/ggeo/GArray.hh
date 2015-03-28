#ifndef GARRAY_H
#define GARRAY_H

#include "GBuffer.hh"

template <class T>
class GArray : public GBuffer {
  public:
     GArray(unsigned int length, const T* values)
     :
       GBuffer(sizeof(T)*length, (void*)values, sizeof(T), 1),
       m_length(length)
     {
     }

     virtual ~GArray()
     {
     }

     unsigned int getLength()
     {
         return m_length ;
     }
     const T* getValues()
     {
         return (T*)m_pointer ;
     }

  private:
     unsigned int m_length ;
     const T* m_values ;

};

#endif


