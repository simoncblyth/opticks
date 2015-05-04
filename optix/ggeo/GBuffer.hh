#ifndef GBUFFER_H
#define GBUFFER_H

class GBuffer {
    public:
      GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem)
         :
         m_nbytes(nbytes),     // total number of bytes 
         m_pointer(pointer),   // pointer to the bytes
         m_itemsize(itemsize), // sizeof each item, eg sizeof(gfloat3) = 3*4 = 12
         m_nelem(nelem)        // number of elements for each item, eg 2 or 3 for floats per vertex
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
      virtual unsigned int getItemSize()
      {
          return m_itemsize ;
      }
      virtual unsigned int getNumElements()
      {
          return m_nelem ;
      }
      virtual unsigned int getNumItems()
      {
          return m_nbytes/m_itemsize ;
      }
      virtual unsigned int getNumElementsTotal()
      {
          return m_nbytes/m_itemsize*m_nelem ;
      }

      /*

          eg 10 float3 vertices, where the item is regarded at the float3 

               NumBytes          10*3*4 = 120 bytes
               ItemSize             3*4 = 12 bytes
               NumElements            3      3 float elements make up the float3
               NumItems              10  =  NumBytes/ItemSize  = 120 bytes/ 12 bytes 
               NumElementsTotal      30  =  NumItems*NumElements = 10*3 

      */ 

      void Summary(const char* msg="GBuffer::Summary");

  protected:
      unsigned int m_nbytes ;
      void*        m_pointer ; 
      unsigned int m_itemsize ;
      unsigned int m_nelem ;

}; 

#endif

