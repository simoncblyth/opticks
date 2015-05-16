#pragma once

class G4StepNPY ; 

#include "uif.h"
#include "numpy.hpp"
#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "string.h"
#include "stdlib.h"
#include "assert.h"

class NPY {
   friend class G4StepNPY ; 

   public:
       static std::string path(const char* typ, const char* tag);
       static NPY* debugload(const char* path);
       static NPY* load(const char* path);
       static NPY* load(const char* typ, const char* tag);
       static NPY* make_vec3(float* m2w, unsigned int npo=100);  
       static NPY* make_vec4(unsigned int npo, float value=0.f);

       // ctor takes ownership of a copy of the inputs 
       NPY(std::vector<int>& shape, double* data, std::string& metadata) ;
       NPY(std::vector<int>& shape, float*  data, std::string& metadata) ;
       NPY(std::vector<int>& shape, std::vector<float>& data, std::string& metadata) ;

       void save(const char* path);
       unsigned int getLength();
       unsigned int getDimensions();
       std::vector<int>& getShapeVector();
       unsigned int getShape(unsigned int dim);
       unsigned int getNumValues(unsigned int from_dim=1);
       unsigned int getNumBytes(unsigned int from_dim=1);
       float* getFloats();
       void* getBytes();
       void read(void* ptr);

    public:
       // methods assuming 3D shape
       unsigned int getFloatIndex(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getByteIndex(unsigned int i, unsigned int j, unsigned int k);
       int          getBufferId();  // either -1 if not uploaded, or the OpenGL buffer Id
       unsigned int getUSum(unsigned int j, unsigned int k);

       float        getFloat(unsigned int i, unsigned int j, unsigned int k);
       unsigned int getUInt( unsigned int i, unsigned int j, unsigned int k);
       void         setFloat(unsigned int i, unsigned int j, unsigned int k, float value);
       void         setUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int value);

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
       std::vector<float> m_data ; 
       std::string        m_metadata ; 

};



inline int NPY::getBufferId()
{
    return m_buffer_id ;
}
inline void NPY::setBufferId(int buffer_id)
{
    m_buffer_id = buffer_id  ;
}





inline unsigned int NPY::getNumValues(unsigned int from_dim)
{
    unsigned int nfloat = 1 ; 
    for(unsigned int i=from_dim ; i < m_shape.size() ; i++) nfloat *= m_shape[i] ;
    return nfloat ;  
}
inline unsigned int NPY::getNumBytes(unsigned int from_dim)
{
    assert(sizeof(float) == 4);
    return sizeof(float)*getNumValues(from_dim);
}
inline unsigned int NPY::getDimensions()
{
    return m_shape.size();
}

inline std::vector<int>& NPY::getShapeVector()
{
    return m_shape ; 
}


inline unsigned int NPY::getShape(unsigned int n)
{
    return n < m_shape.size() ? m_shape[n] : -1 ;
}
inline unsigned int NPY::getLength()
{
    return getShape(0);
}


inline float* NPY::getFloats()
{
    return m_data.data();
}
inline void* NPY::getBytes()
{
    return (void*)getFloats();
}


inline unsigned int NPY::getByteIndex(unsigned int i, unsigned int j, unsigned int k)
{
    return sizeof(float)*getFloatIndex(i,j,k);
}
inline unsigned int NPY::getFloatIndex(unsigned int i, unsigned int j, unsigned int k)
{
    assert(m_dim == 3 ); 
    unsigned int nj = m_len1 ;
    unsigned int nk = m_len2 ;
    return  i*nj*nk + j*nk + k ;
}

inline float NPY::getFloat(unsigned int i, unsigned int j, unsigned int k)
{
    unsigned int idx = getFloatIndex(i,j,k);
    float* data = getFloats();
    return  *(data + idx);
}
inline void NPY::setFloat(unsigned int i, unsigned int j, unsigned int k, float value)
{
    unsigned int idx = getFloatIndex(i,j,k);
    float* data = getFloats();
    *(data + idx) = value ;
}
inline unsigned int NPY::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    uif_t uif ; 
    uif.f = getFloat(i,j,k);
    return uif.u ;
}
inline void NPY::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    uif_t uif ; 
    uif.u = value ;
    setFloat(i,j,k,uif.f); 
}





