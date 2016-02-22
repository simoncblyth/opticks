#pragma once

template <typename T> class NPY ;

class GCSG {

    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              NJ = 4,
              NK = 4
            } ;

    public:
        GCSG(NPY<float>* buffer);
        NPY<float>* getCSGBuffer();
        void dump(const char* msg="GCSG::dump");

    public:
        unsigned int getNumItems();
    public:
        unsigned int getTypeCode(unsigned int i);
    public:
        unsigned int getIndex(unsigned int i);
        unsigned int getNumChild(unsigned int i);
        unsigned int getFirstChildIndex(unsigned int i);
        unsigned int getLastChildIndex(unsigned int i);
    private:
        unsigned int getUInt(unsigned int i, unsigned int j, unsigned int k);

    private:
        NPY<float>*        m_csg_buffer ; 
};


inline GCSG::GCSG(NPY<float>* buffer) 
      :
      m_csg_buffer(buffer)
{
}
      
inline NPY<float>* GCSG::getCSGBuffer()
{
    return m_csg_buffer ; 
}



inline unsigned int GCSG::getTypeCode(unsigned int i)
{
    return getUInt(i, 2, 0);
}
inline unsigned int GCSG::getIndex(unsigned int i)
{
    return getUInt(i, 3, 0);
}
inline unsigned int GCSG::getNumChild(unsigned int i)
{
    return getUInt(i, 3, 1);
}
inline unsigned int GCSG::getFirstChildIndex(unsigned int i)
{
    return getUInt(i, 3, 2);
}
inline unsigned int GCSG::getLastChildIndex(unsigned int i)
{
    return getUInt(i, 3, 3);
}



