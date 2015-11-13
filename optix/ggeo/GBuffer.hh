#pragma once

#include <cassert>

/*
eg 10 float3 vertices, where the item is regarded at the float3 

   NumBytes          10*3*4 = 120 bytes
   ItemSize             3*4 = 12 bytes
   NumElements            3      3 float elements make up the float3
   NumItems              10  =  NumBytes/ItemSize  = 120 bytes/ 12 bytes 
   NumElementsTotal      30  =  NumItems*NumElements = 10*3 
*/ 


struct NSlice ; 

// ARE TRANSITIONING FROM GBuffer TO NPY<T> WHERE POSSIBLE : 
//    **DO NOT USE GBuffer IN NEW DEVELOPMENTS**

class GBuffer {
    public:
        GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem);
    public:
        void reshape(unsigned int nelem);
        // NB reshape just changes interpretation, there is no change to NumBytes or NumElementsTotal
        //    only NumItems and NumElements are changed (reversibly)
    public:
        unsigned int getNumBytes();
        void*        getPointer();
        unsigned int getItemSize();
        unsigned int getNumElements();
        unsigned int getNumItems();
        unsigned int getNumElementsTotal();
    public:
        bool isEqual(GBuffer* other);
        float fractionDifferent(GBuffer* other);
        void Summary(const char* msg="GBuffer::Summary");
    public:
        GBuffer* make_slice(const char* slice); 
        GBuffer* make_slice(NSlice* slice); 
    public:
        template<typename T>
        void dump(const char* msg="GBuffer::dump", unsigned int limit=10);

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
         m_buffer_id(-1),       // OpenGL buffer Id, set by Renderer on uploading to GPU 
         m_buffer_target(0)
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

inline void GBuffer::reshape(unsigned int nelem)
{
    if(nelem == m_nelem) return ; 

    bool up = nelem > m_nelem ; 
    if(up) 
    { 
        // reinterpret to a larger "item" with more elements
        assert(nelem % m_nelem == 0);
        unsigned int factor = nelem/m_nelem  ;
        m_nelem = nelem ;
        m_itemsize = m_itemsize*factor ; 
    }
    else
    { 
        // reinterpret to a smaller "item" with less elements  
        assert(m_nelem % nelem == 0);
        unsigned int factor = m_nelem/nelem  ;
        m_nelem = nelem ;
        m_itemsize = m_itemsize/factor ; 
    }
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



