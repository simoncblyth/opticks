#pragma once

class G4StepNPY ; 

#include "uif.h"
#include "ucharfour.h"
#include "charfour.h"

#include "numpy.hpp"
#include <vector>
#include <string>
#include <set>
#include <map>

#include <cstring>
#include <cstdlib>
#include <cassert>

#include "NPYBase.hpp"

struct NSlice ; 


/*
Interop NumPy -> NPY
======================

The type of the NumPy array saved from python
needs to match the NPY basis type 
eg NPY<float> NPY<short> NPY<double>

From python control the type, eg to save as float with::

    In [3]: a.dtype
    Out[3]: dtype('float64')

    In [4]: b = np.array(a, dtype=np.float32)

    np.save("/tmp/slowcomponent.npy", b ) 


Going the other way, NPY -> NumPy just works as NumPy 
recognises the type on loading, thanks to the numpy.hpp metadata header.

*/


template <class T>
class NPY : public NPYBase {

   friend class AxisNPY ; 
   friend class SequenceNPY ; 
   friend class RecordsNPY ; 
   friend class PhotonsNPY ; 
   friend class HitsNPY ; 
   friend class G4StepNPY ; 
   friend class MaterialLibNPY ; 

   public:
       static Type_t type ;  // for type branching 
       static T UNSET ; 

   public:
       // NB favor vec4 over vec3 for better GPU performance (due to memory coalescing/alignment)
       static NPY<T>* make_vec3(float* m2w, unsigned int npo=100);  

       static NPY<T>* make(std::vector<int>& shape);
       static NPY<T>* make(unsigned int ni);
       static NPY<T>* make(unsigned int ni, unsigned int nj );
       static NPY<T>* make(unsigned int ni, unsigned int nj, unsigned int nk );
       static NPY<T>* make(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl );

       static NPY<T>* make_modulo(NPY<T>* src, unsigned int scaledown);

       // ctor takes ownership of a copy of the inputs 
       NPY(std::vector<int>& shape, T*  data            , std::string& metadata) ;
       NPY(std::vector<int>& shape, std::vector<T>& data, std::string& metadata) ;

   public:
   public:
       static NPY<T>* debugload(const char* path);
       static NPY<T>* load(const char* path);
       static NPY<T>* load(const char* dir, const char* name);
       static NPY<T>* load(const char* typ, const char* tag, const char* det);
       static NPY<T>* load(const char* pfx, const char* gen, const char* tag, const char* det); // type split into prefix and gen:cerenkov,scintillation,torch
 
       void save(const char* path);
       void save(const char* dir, const char* name);
       void save(const char* typ, const char* tag, const char* det );
       void save(const char* tfmt, const char* targ, const char* tag, const char* det);

       bool exists(const char* path);
       bool exists(const char* dir, const char* name);
       bool exists(const char* typ, const char* tag, const char* det);
       bool exists(const char* tfmt, const char* targ, const char* tag, const char* det );

       // manipulations change types, not tags:  tfmt % targ -> typ
   public:
       NPY<T>* make_slice(const char* slice);
       NPY<T>* make_slice(NSlice* slice);
   public:
       T maxdiff(NPY<T>* other);
   public:
       T* getValues();
       T* getValues(unsigned int i);
       void* getBytes();
       void* getPointer();   // aping GBuffer for easier migration
       void read(void* ptr);
       T* grow(unsigned int nitems); // increase size to contain an extra nitems, return pointer to start of them
    public:
       void add(NPY<T>* other);      // add another buffer, it must have same itemsize (ie size after 1st dimension)
       void add(const T* values, unsigned int nvals);   // add values, nvals must be integral multiple of the itemsize  
       void add(void* bytes, unsigned int nbytes); // add bytes,  nbytes must be integral multiple of itemsize in bytes
       void add(T x, T y, T z, T w) ;   // add values of a quad, itemsize must be 4 
    public:
       std::vector<T>& data();
       void setData(T* data);
       void fill(T value);
       T* zero();
       T* allocate();
    public:
       T* getUnsetItem();
       bool isUnsetItem(unsigned int i);
    public:
       void dump(const char* msg="NPY::dump", unsigned int limit=15);
    public:
       // methods assuming 3D shape
       std::set<int> uniquei(unsigned int j, unsigned int k);

       std::map<int,int> count_uniquei(unsigned int j, unsigned int k, int sj=-1, int sk=-1);
       // when both sj and sk are >-1 the float specified is used 
       // to sign the boundary code used in the map 

       std::map<unsigned int,unsigned int> count_unique_u(unsigned int j, unsigned int k);

       std::vector<std::pair<int,int> > count_uniquei_descending(unsigned int j, unsigned int k); 
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);

    public:
       // type shifting get/set using union trick

       unsigned int getUSum(unsigned int j, unsigned int k);

       T            getValue(unsigned int i, unsigned int j, unsigned int k);
       float        getFloat(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getUInt( unsigned int i, unsigned int j, unsigned int k);
       int          getInt(  unsigned int i, unsigned int j, unsigned int k);
       void         getU( short& value, unsigned short& uvalue, unsigned char& msb, unsigned char& lsb, unsigned int i, unsigned int j, unsigned int k);
       ucharfour    getUChar4( unsigned int i, unsigned int j, unsigned int k0, unsigned int k1 );
       charfour     getChar4( unsigned int i, unsigned int j, unsigned int k0, unsigned int k1 );


       void         setValue(unsigned int i, unsigned int j, unsigned int k, T value);
       void         setFloat(unsigned int i, unsigned int j, unsigned int k, float value);
       void         setUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int value);
       void         setInt(  unsigned int i, unsigned int j, unsigned int k, int value);

       void         setQuad(unsigned int i, unsigned int j, glm::vec4&  vec );
       void         setQuadI(unsigned int i, unsigned int j, glm::ivec4& vec );
       void         setQuadU(unsigned int i, unsigned int j, glm::uvec4& vec );

       void         setQuad(unsigned int i, unsigned int j, glm::ivec4& vec );
       void         setQuad(unsigned int i, unsigned int j, glm::uvec4& vec );

       void         setQuad(unsigned int i, unsigned int j, float x, float y=0.f, float z=0.f, float w=0.f );
       glm::vec4    getQuad(unsigned int i, unsigned int j );
       glm::ivec4   getQuadI(unsigned int i, unsigned int j );
       glm::uvec4   getQuadU(unsigned int i, unsigned int j );

   //private:
   public:
       std::vector<T>     m_data ; 
       T*                 m_unset_item ; 
 
};




template <typename T> 
inline std::vector<T>& NPY<T>::data()
{
    return m_data ;
}


template <typename T> 
inline T* NPY<T>::getValues()
{
    return m_data.data();
}

template <typename T> 
inline T* NPY<T>::getValues(unsigned int i)
{
    unsigned int idx = getValueIndex(i,0,0);
    return m_data.data() + idx ;
}


template <typename T> 
inline void* NPY<T>::getBytes()
{
    return hasData() ? (void*)getValues() : NULL ;
}

template <typename T> 
inline void* NPY<T>::getPointer()
{
    return getBytes() ;
}




template <typename T> 
inline T NPY<T>::getValue(unsigned int i, unsigned int j, unsigned int k)
{
    unsigned int idx = getValueIndex(i,j,k);
    T* data = getValues();
    return  *(data + idx);
}

template <typename T> 
inline void NPY<T>::getU( short& value, unsigned short& uvalue, unsigned char& msb, unsigned char& lsb, unsigned int i, unsigned int j, unsigned int k)
{
    // used for unpacking photon records

    assert(type == SHORT); // pragmatic template specialization, by death if you try to use the wrong one...

    unsigned int index = getValueIndex(i,j,k);

    value = m_data[index] ;

    hui_t hui ;
    hui.short_ = value ; 

    uvalue = hui.ushort_ ;

    msb = ( uvalue & 0xFF00 ) >> 8  ;  

    lsb = ( uvalue & 0xFF ) ;  
}

template <typename T> 
ucharfour  NPY<T>::getUChar4( unsigned int i, unsigned int j, unsigned int k0, unsigned int k1 )
{
    assert(type == SHORT); // OOPS: pragmatic template specialization, by death if you try to use the wrong one... 

    unsigned int index_0 = getValueIndex(i,j,k0);
    unsigned int index_1 = getValueIndex(i,j,k1);

    hui_t hui_0, hui_1 ;

    hui_0.short_ = m_data[index_0];
    hui_1.short_ = m_data[index_1];

    ucharfour v ; 

    v.x = (hui_0.ushort_ & 0xFF ) ; 
    v.y = (hui_0.ushort_ & 0xFF00 ) >> 8 ; 
    v.z = (hui_1.ushort_ & 0xFF ) ; 
    v.w = (hui_1.ushort_ & 0xFF00 ) >> 8 ; 

    return v ;
}

template <typename T> 
charfour  NPY<T>::getChar4( unsigned int i, unsigned int j, unsigned int k0, unsigned int k1 )
{
    assert(type == SHORT); // OOPS: pragmatic template specialization, by death if you try to use the wrong one... 

    unsigned int index_0 = getValueIndex(i,j,k0);
    unsigned int index_1 = getValueIndex(i,j,k1);

    hui_t hui_0, hui_1 ;

    hui_0.short_ = m_data[index_0];
    hui_1.short_ = m_data[index_1];

    charfour v ; 

    v.x = (hui_0.short_ & 0xFF ) ; 
    v.y = (hui_0.short_ & 0xFF00 ) >> 8 ; 
    v.z = (hui_1.short_ & 0xFF ) ; 
    v.w = (hui_1.short_ & 0xFF00 ) >> 8 ; 

    // hmm signbit complications ?
    return v ;
}





template <typename T> 
inline void NPY<T>::setValue(unsigned int i, unsigned int j, unsigned int k, T value)
{
    unsigned int idx = getValueIndex(i,j,k);
    T* dat = getValues();
    *(dat + idx) = value ;
}


template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, glm::vec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setValue(i,j,k,vec[k]); 
}


template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, float x, float y, float z, float w )
{
    glm::vec4 vec(x,y,z,w); 
    setQuad(i, j, vec);
}


template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, glm::ivec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setValue(i,j,k,vec[k]); 
}

template <typename T> 
inline void NPY<T>::setQuadI(unsigned int i, unsigned int j, glm::ivec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setInt(i,j,k,vec[k]); 
}
template <typename T> 
inline void NPY<T>::setQuadU(unsigned int i, unsigned int j, glm::uvec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setUInt(i,j,k,vec[k]); 
}



template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, glm::uvec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setValue(i,j,k,vec[k]); 
}




template <typename T> 
inline glm::vec4 NPY<T>::getQuad(unsigned int i, unsigned int j)
{
    assert( m_len2 == 4 );  
    glm::vec4 vec ; 
    for(unsigned int k=0 ; k < 4 ; k++) vec[k] = getValue(i,j,k); 
    return vec ; 
}

template <typename T> 
inline glm::ivec4 NPY<T>::getQuadI(unsigned int i, unsigned int j)
{
    assert( m_len2 == 4 );  
    glm::ivec4 vec ; 
    for(unsigned int k=0 ; k < 4 ; k++) vec[k] = getValue(i,j,k); 
    return vec ; 
}

template <typename T> 
inline glm::uvec4 NPY<T>::getQuadU(unsigned int i, unsigned int j)
{
    assert( m_len2 == 4 );  
    glm::uvec4 vec ; 
    for(unsigned int k=0 ; k < 4 ; k++) vec[k] = getUInt(i,j,k); 
    return vec ; 
}





// type shifting get/set using union trick


template <typename T> 
inline float NPY<T>::getFloat(unsigned int i, unsigned int j, unsigned int k)
{
    uif_t uif ; 

    T t = getValue(i,j,k);
    switch(type)
    {
        case FLOAT:uif.f = t ; break ; 
        case DOUBLE:uif.f = t ; break ; 
        case SHORT:uif.i = t ; break ; 
        default: assert(0);  break ;
    }
    return uif.f ;
}

template <typename T> 
inline void NPY<T>::setFloat(unsigned int i, unsigned int j, unsigned int k, float  value)
{
    uif_t uif ; 
    uif.f = value ;

    T t ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,t); 
}



template <typename T> 
inline unsigned int NPY<T>::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    uif_t uif ; 

    T t = getValue(i,j,k);
    switch(type)
    {
        case FLOAT:uif.f = t ; break ; 
        case DOUBLE:uif.f = t ; break ; 
        case SHORT:uif.i = t ; break ; 
        case UINT:uif.u = t ; break ; 
        case INT:uif.i = t ; break ; 
        default: assert(0);  break ;
    }
    return uif.u ;
}

template <typename T> 
inline void NPY<T>::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    uif_t uif ; 
    uif.u = value ;

    T t ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        case UINT:t = uif.u ; break ; 
        case INT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,t); 
}

template <typename T> 
inline int NPY<T>::getInt(unsigned int i, unsigned int j, unsigned int k)
{
    uif_t uif ;             // how does union handle different sizes ? 
    T t = getValue(i,j,k);
    switch(type)
    {   
        case FLOAT: uif.f = t ; break;
        case DOUBLE: uif.f = t ; break;
        case SHORT: uif.i = t ; break;
        default: assert(0);   break;
    }
    return uif.i ;
}

template <typename T> 
inline void NPY<T>::setInt(unsigned int i, unsigned int j, unsigned int k, int value)
{
    uif_t uif ; 
    uif.i = value ;

    T t ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,t); 
}




