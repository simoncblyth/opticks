#ifndef ARRAY_H
#define ARRAY_H

#include "Buffer.hh"

template <class T>
class Array : public Buffer {
  public:
     Array(unsigned int length, const T* values)
     :
       Buffer(sizeof(T)*length, (void*)values),
       m_length(length)
     {
     }

     virtual ~Array()
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


