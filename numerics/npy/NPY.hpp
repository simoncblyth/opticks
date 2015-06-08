#pragma once

class G4StepNPY ; 

#include "uif.h"
#include "numpy.hpp"
#include <vector>
#include <string>
#include <set>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "string.h"
#include "stdlib.h"
#include "assert.h"






template <class T>
class NPY {
   friend class PhotonsNPY ; 
   friend class G4StepNPY ; 

   public:
       static const char type ;  // for type branching 
       enum { FLOAT, SHORT, DOUBLE };

   public:
       static NPY<T>* make_vec3(float* m2w, unsigned int npo=100);  
       static NPY<T>* make_vec4(unsigned int ni, unsigned int nj=1, T value=0);

       // ctor takes ownership of a copy of the inputs 
       NPY(std::vector<int>& shape, T*  data            , std::string& metadata) ;
       NPY(std::vector<int>& shape, std::vector<T>& data, std::string& metadata) ;

   public:
       static std::string path(const char* typ, const char* tag);
       static NPY* debugload(const char* path);
       static NPY* load(const char* path);
       static NPY* load(const char* typ, const char* tag);
 
       void save(const char* path);
       void save(const char* typ, const char* tag);
       void save(const char* tfmt, const char* targ, const char* tag );
       // manipulations typically change types, not tags:  tfmt % targ -> typ

   public:
       unsigned int getLength();
       unsigned int getDimensions();
       std::vector<int>& getShapeVector();
       unsigned int getShape(unsigned int dim);
       unsigned int getNumValues(unsigned int from_dim=1);
       unsigned int getNumBytes(unsigned int from_dim=1);
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
       // methods assuming 3D shape
       unsigned int getValueIndex(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getByteIndex(unsigned int i, unsigned int j, unsigned int k);
       int          getBufferId();  // either -1 if not uploaded, or the OpenGL buffer Id

       unsigned int getUSum(unsigned int j, unsigned int k);

       T            getValue(unsigned int i, unsigned int j, unsigned int k);
       float        getFloat(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getUInt( unsigned int i, unsigned int j, unsigned int k);
       int          getInt(  unsigned int i, unsigned int j, unsigned int k);

       void         setValue(unsigned int i, unsigned int j, unsigned int k, T value);
       void         setFloat(unsigned int i, unsigned int j, unsigned int k, float value);
       void         setUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int value);
       void         setInt(  unsigned int i, unsigned int j, unsigned int k, int value);

    public:
       std::string  getItemShape(unsigned int ifr=1);
       void         setBufferId(int buffer_id);
       std::string description(const char* msg);
       void Summary(const char* msg="NPY::Summary");
       void debugdump();

   protected:
       unsigned int       m_dim ; 
       unsigned int       m_len0 ; 
       unsigned int       m_len1 ; 
       unsigned int       m_len2 ; 
       int                m_buffer_id ; 

   private:
       std::vector<int>   m_shape ; 
       std::vector<T>     m_data ; 
       std::string        m_metadata ; 

};




template <typename T> 
inline int NPY<T>::getBufferId()
{
    return m_buffer_id ;
}

template <typename T> 
inline void NPY<T>::setBufferId(int buffer_id)
{
    m_buffer_id = buffer_id  ;
}




template <typename T> 
inline unsigned int NPY<T>::getNumValues(unsigned int from_dim)
{
    unsigned int nvals = 1 ; 
    for(unsigned int i=from_dim ; i < m_shape.size() ; i++) nvals *= m_shape[i] ;
    return nvals ;  
}
template <typename T> 
inline unsigned int NPY<T>::getNumBytes(unsigned int from_dim)
{
    return sizeof(T)*getNumValues(from_dim);
}
template <typename T> 
inline unsigned int NPY<T>::getDimensions()
{
    return m_shape.size();
}
template <typename T> 
inline std::vector<int>& NPY<T>::getShapeVector()
{
    return m_shape ; 
}

template <typename T> 
inline unsigned int NPY<T>::getShape(unsigned int n)
{
    return n < m_shape.size() ? m_shape[n] : -1 ;
}
template <typename T> 
inline unsigned int NPY<T>::getLength()
{
    return getShape(0);
}

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
inline unsigned int NPY<T>::getByteIndex(unsigned int i, unsigned int j, unsigned int k)
{
    return sizeof(T)*getValueIndex(i,j,k);
}

template <typename T> 
inline unsigned int NPY<T>::getValueIndex(unsigned int i, unsigned int j, unsigned int k)
{
    assert(m_dim == 3 ); 
    unsigned int nj = m_len1 ;
    unsigned int nk = m_len2 ;
    return  i*nj*nk + j*nk + k ;
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





