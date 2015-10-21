#pragma once

/*
eg 10 float3 vertices, where the item is regarded at the float3 

   NumBytes          10*3*4 = 120 bytes
   ItemSize             3*4 = 12 bytes
   NumElements            3      3 float elements make up the float3
   NumItems              10  =  NumBytes/ItemSize  = 120 bytes/ 12 bytes 
   NumElementsTotal      30  =  NumItems*NumElements = 10*3 
*/ 


struct NSlice ; 

class GBuffer {
    public:
        GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem);
    public:
        unsigned int getNumBytes();
        void* getPointer();
        unsigned int getItemSize();
        unsigned int getNumElements();
        unsigned int getNumItems();
        unsigned int getNumElementsTotal();
    public:
        bool isEqual(GBuffer* other);
        float fractionDifferent(GBuffer* other);
        void Summary(const char* msg="GBuffer::Summary");
        void dump(const char* msg="GBuffer::dump", unsigned int nfold=4);
    public:
        GBuffer* make_slice(const char* slice); 
        GBuffer* make_slice(NSlice* slice); 
    public:
        template<typename T>
        void save(const char* path);

        template<typename T>
        static GBuffer* load(const char* path);

        template<typename T>
        static GBuffer* load(const char* dir, const char* name);

    public:
        // OpenGL related : but not requiring any headers
        void         setBufferId(int buffer_id);
        int          getBufferId();  // either -1 if not uploaded, or the OpenGL buffer Id
        void         setBufferTarget(int buffer_target);
        int          getBufferTarget();

    protected:
         unsigned int m_nbytes ;
         void*        m_pointer ; 
         unsigned int m_itemsize ;
         unsigned int m_nelem ;
    private:
         int          m_buffer_id ; 
         int          m_buffer_target ; 

}; 



inline GBuffer::GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem)
         :
         m_nbytes(nbytes),     // total number of bytes 
         m_pointer(pointer),   // pointer to the bytes
         m_itemsize(itemsize), // sizeof each item, eg sizeof(gfloat3) = 3*4 = 12
         m_nelem(nelem),       // number of elements for each item, eg 2 or 3 for floats per vertex or 16 for a 4x4 matrix
         m_buffer_id(-1)       // OpenGL buffer Id, set by Renderer on uploading to GPU 
{
}


inline unsigned int GBuffer::getNumBytes()
{
    return m_nbytes ;
}
inline void* GBuffer::getPointer()
{
    return m_pointer ;
}
inline unsigned int GBuffer::getItemSize()
{
    return m_itemsize ;
}
inline unsigned int GBuffer::getNumElements()
{
    return m_nelem ;
}
inline unsigned int GBuffer::getNumItems()
{
    return m_nbytes/m_itemsize ;
}
inline unsigned int GBuffer::getNumElementsTotal()
{
    return m_nbytes/m_itemsize*m_nelem ;
}


// OpenGL related
inline void GBuffer::setBufferId(int buffer_id)
{
    m_buffer_id = buffer_id  ;
}
inline int GBuffer::getBufferId()
{
    return m_buffer_id ;
}
inline void GBuffer::setBufferTarget(int buffer_target)
{
    m_buffer_target = buffer_target  ;
}
inline int GBuffer::getBufferTarget()
{
    return m_buffer_target ;
}



