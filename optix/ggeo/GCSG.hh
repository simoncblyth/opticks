#pragma once

template <typename T> class NPY ;
class GItemList ; 

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
        GCSG(NPY<float>* buffer, GItemList* materials, GItemList* lvnames, GItemList* pvnames);

        NPY<float>* getCSGBuffer();
        void dump(const char* msg="GCSG::dump");
    public:
        const char* getMaterialName(unsigned int nodeindex);
        const char* getLVName(unsigned int nodeindex);
        const char* getPVName(unsigned int nodeindex);
    public:
        unsigned int getNumItems();
    public:
        float getX(unsigned int i);
        float getY(unsigned int i);
        float getZ(unsigned int i);
        float getOuterRadius(unsigned int i);
        float getInnerRadius(unsigned int i);
        float getSizeZ(unsigned int i);
        float getStartTheta(unsigned int i);
        float getDeltaTheta(unsigned int i);
    public:
        unsigned int getTypeCode(unsigned int i);
        bool isUnion(unsigned int i);
        bool isIntersection(unsigned int i);
        bool isSphere(unsigned int i);
        bool isTubs(unsigned int i);

        unsigned int getNodeIndex(unsigned int i);  // 1-based index, 0:unset
        unsigned int getParentIndex(unsigned int i);  // 1-based index, 0:unset

        const char* getTypeName(unsigned int i);
    public:
        unsigned int getIndex(unsigned int i);
        unsigned int getNumChildren(unsigned int i);
        unsigned int getFirstChildIndex(unsigned int i);
        unsigned int getLastChildIndex(unsigned int i);
    private:
        float        getFloat(unsigned int i, unsigned int j, unsigned int k);
        unsigned int getUInt(unsigned int i, unsigned int j, unsigned int k);

    private:
        NPY<float>*        m_csg_buffer ; 
        GItemList*         m_materials ; 
        GItemList*         m_lvnames ; 
        GItemList*         m_pvnames ; 
};


inline GCSG::GCSG(NPY<float>* buffer, GItemList* materials, GItemList* lvnames, GItemList* pvnames) 
      :
      m_csg_buffer(buffer),
      m_materials(materials),
      m_lvnames(lvnames),
      m_pvnames(pvnames)

{
}
      
inline NPY<float>* GCSG::getCSGBuffer()
{
    return m_csg_buffer ; 
}



inline bool GCSG::isUnion(unsigned int i)
{
    return getTypeCode(i) == UNION ; 
}
inline bool GCSG::isIntersection(unsigned int i)
{
    return getTypeCode(i) == INTERSECTION ; 
}
inline bool GCSG::isSphere(unsigned int i)
{
    return getTypeCode(i) == SPHERE ; 
}
inline bool GCSG::isTubs(unsigned int i)
{
    return getTypeCode(i) == TUBS ; 
}





inline const char* GCSG::getTypeName(unsigned int i)
{
    unsigned int tc = getTypeCode(i);
    return TypeName(tc) ;
}




inline float GCSG::getX(unsigned int i){            return getFloat(i, 0, 0 ); }
inline float GCSG::getY(unsigned int i){            return getFloat(i, 0, 1 ); }
inline float GCSG::getZ(unsigned int i){            return getFloat(i, 0, 2 ); }
inline float GCSG::getOuterRadius(unsigned int i){  return getFloat(i, 0, 3 ); }

inline float GCSG::getStartTheta(unsigned int i) {  return getFloat(i, 1, 0 ); }
inline float GCSG::getDeltaTheta(unsigned int i) {  return getFloat(i, 1, 1 ); }
inline float GCSG::getSizeZ(unsigned int i) {       return getFloat(i, 1, 2 ); }
inline float GCSG::getInnerRadius(unsigned int i) { return getFloat(i, 1, 3 ); }

inline unsigned int GCSG::getTypeCode(unsigned int i){         return getUInt(i, 2, 0); }
inline unsigned int GCSG::getNodeIndex(unsigned int i) {       return getUInt(i, 2, 1); }
inline unsigned int GCSG::getParentIndex(unsigned int i) {     return getUInt(i, 2, 2); }

inline unsigned int GCSG::getIndex(unsigned int i) {           return getUInt(i, 3, 0); }
inline unsigned int GCSG::getNumChildren(unsigned int i) {     return getUInt(i, 3, 1); }
inline unsigned int GCSG::getFirstChildIndex(unsigned int i) { return getUInt(i, 3, 2); }
inline unsigned int GCSG::getLastChildIndex(unsigned int i) {  return getUInt(i, 3, 3); }





