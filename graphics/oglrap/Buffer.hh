#ifndef BUFFER_H
#define BUFFER_H

class Buffer {
  public:
      Buffer(unsigned int nbytes, void* pointer)
         :
         m_nbytes(nbytes),
         m_pointer(pointer)
      {
      }
      virtual ~Buffer()
      {
      }
      unsigned int getNumBytes()
      {
          return m_nbytes ;
      }
      void* getPointer()
      {
          return m_pointer ;
      }

      void Summary(const char* msg="Buffer::Summary");


  protected:
      unsigned int m_nbytes ;
      void*       m_pointer ; 

}; 

#endif

