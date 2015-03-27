#ifndef GBUFFER_H
#define GBUFFER_H

class GBuffer {
  public:
      GBuffer(unsigned int nbytes, void* pointer)
         :
         m_nbytes(nbytes),
         m_pointer(pointer)
      {
      }
      virtual ~GBuffer()
      {
      }
      virtual unsigned int getNumBytes()
      {
          return m_nbytes ;
      }
      virtual void* getPointer()
      {
          return m_pointer ;
      }

      void Summary(const char* msg="GBuffer::Summary");


  protected:
      unsigned int m_nbytes ;
      void*       m_pointer ; 

}; 

#endif

