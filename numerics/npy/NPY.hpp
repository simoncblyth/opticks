#pragma once

class G4StepNPY ; 

#include "uif.h"
#include "numpy.hpp"
#include <vector>
#include <string>
#include <set>
#include <map>

#include "string.h"
#include "stdlib.h"
#include "assert.h"

#include "NPYBase.hpp"


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
   friend class PhotonsNPY ; 
   friend class G4StepNPY ; 

   public:
       static Type_t type ;  // for type branching 

   public:
       static NPY<T>* make_vec3(float* m2w, unsigned int npo=100);  
       static NPY<T>* make_vec4(unsigned int ni, unsigned int nj=1, T value=0);

       // ctor takes ownership of a copy of the inputs 
       NPY(std::vector<int>& shape, T*  data            , std::string& metadata) ;
       NPY(std::vector<int>& shape, std::vector<T>& data, std::string& metadata) ;

   public:
       static NPY<T>* debugload(const char* path);
       static NPY<T>* load(const char* path);
       static NPY<T>* load(const char* typ, const char* tag);
 
       void save(const char* path);
       void save(const char* typ, const char* tag);
       void save(const char* tfmt, const char* targ, const char* tag );
       // manipulations change types, not tags:  tfmt % targ -> typ

   public:
       T* getValues();
       void* getBytes();
       void read(void* ptr);

    public:
       // methods assuming 3D shape
       std::set<int> uniquei(unsigned int j, unsigned int k);

       std::map<int,int> count_uniquei(unsigned int j, unsigned int k, int sj=-1, int sk=-1);
       // when both sj and sk are >-1 the float specified is used 
       // to sign the boundary code used in the map 

    public:
       // type shifting get/set using union trick

       unsigned int getUSum(unsigned int j, unsigned int k);

       T            getValue(unsigned int i, unsigned int j, unsigned int k);
       float        getFloat(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getUInt( unsigned int i, unsigned int j, unsigned int k);
       int          getInt(  unsigned int i, unsigned int j, unsigned int k);

       void         setValue(unsigned int i, unsigned int j, unsigned int k, T value);
       void         setFloat(unsigned int i, unsigned int j, unsigned int k, float value);
       void         setUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int value);
       void         setInt(  unsigned int i, unsigned int j, unsigned int k, int value);

       void         setQuad(unsigned int i, unsigned int j, glm::vec4&  vec );
       void         setQuad(unsigned int i, unsigned int j, glm::ivec4& vec );

   private:
       std::vector<T>     m_data ; 
 
};






template <typename T> 
inline T* NPY<T>::getValues()
{
    return m_data.data();
}

template <typename T> 
inline void* NPY<T>::getBytes()
{
    return (void*)getValues();
}

template <typename T> 
inline T NPY<T>::getValue(unsigned int i, unsigned int j, unsigned int k)
{
    unsigned int idx = getValueIndex(i,j,k);
    T* data = getValues();
    return  *(data + idx);
}


template <typename T> 
inline void NPY<T>::setValue(unsigned int i, unsigned int j, unsigned int k, T value)
{
    unsigned int idx = getValueIndex(i,j,k);
    T* data = getValues();
    *(data + idx) = value ;
}


template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, glm::vec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setValue(i,j,k,vec[k]); 
}

template <typename T> 
inline void NPY<T>::setQuad(unsigned int i, unsigned int j, glm::ivec4& vec )
{
    assert( m_len2 == 4 );  
    for(unsigned int k=0 ; k < 4 ; k++) setValue(i,j,k,vec[k]); 
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





