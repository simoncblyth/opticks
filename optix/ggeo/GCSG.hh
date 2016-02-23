#pragma once

template <typename T> class NPY ;

class GCSG {
    public:
       // shapes with analytic intersection implementations 
       static const char* SPHERE_ ;
       static const char* TUBS_ ; 
       static const char* UNION_ ; 
       static const char* INTERSECTION_ ; 
       static const char* TypeName(unsigned int typecode);


    public:
        // buffer layout, must match locations in pmt-/csg.py
        enum { 
              NJ = 4,
              NK = 4,
              UNION = 10,  
              INTERSECTION = 20,
              SPHERE = 3,
              TUBS = 4
            } ;

    public:
        GCSG(NPY<float>* buffer);
        NPY<float>* getCSGBuffer();
        void dump(const char* msg="GCSG::dump");

    public:
        unsigned int getNumItems();
    public:
        unsigned int getTypeCode(unsigned int i);
        const char* getTypeName(unsigned int i);
    public:
        unsigned int getIndex(unsigned int i);
        unsigned int getNumChildren(unsigned int i);
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
inline const char* GCSG::getTypeName(unsigned int i)
{
    unsigned int tc = getTypeCode(i);
    return TypeName(tc) ;
}

inline unsigned int GCSG::getIndex(unsigned int i)
{
    return getUInt(i, 3, 0);
}
inline unsigned int GCSG::getNumChildren(unsigned int i)
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



