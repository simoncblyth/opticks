#ifndef ARRAY_H
#define ARRAY_H

#include <GL/glew.h>

template <class T>
class Array {
  public:
     Array(unsigned int length, const T* values)
     :
       m_length(length),
       m_values(values)
     {
     }

     virtual ~Array()
     {
     }

     unsigned int getNumBytes()
     {
         return sizeof(T)*m_length ;
     }
     unsigned int getLength()
     {
         return m_length ;
     }
     const T* getValues()
     {
         return m_values ;
     }

     void upload(GLenum target)
     {
         glGenBuffers(1, &m_id);
         glBindBuffer(target, m_id);
         glBufferData(target, sizeof(T)*m_length, m_values, GL_STATIC_DRAW);
     }
     GLuint getId()
     {
         return m_id ; 
     }

  private:
     unsigned int m_length ;
     const T* m_values ;
     GLuint   m_id ;

};

#endif


