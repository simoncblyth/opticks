#pragma once

class NPY ; 
#include <glm/glm.hpp>

/*

VecNPY is ultra-lightweight, just managing 

* pointer to data
* parameters for addressing the data (stride, offset, count)
* characteristics of the data 
  (high, low, center, dimensions, extent, model2world matrix)


* Considered calling this SliceNPY but this 
  is much less ambitious than NumPy slicing 
  so stick with VecNPY to express the intended simplicity

* Hmm, maybe ViewNPY is better name.

*/

class VecNPY {
    public:
        //
        // Ctor assumes 3-dimensional NPY array structure 
        // with shapes like (10000,6,4) in which case j 0:5 k 0:3 size=1:4 (usually 4 as quads are most efficient)
        //
        VecNPY(const char* name, NPY* npy, unsigned int j, unsigned int k, unsigned int size=4, char type='f') ;

    public:
        void dump(const char* msg);
        void Summary(const char* msg);
        void Print(const char* msg);

        NPY*         getNPY(){    return m_npy ; }
        void*        getBytes(){  return m_bytes ; }
        unsigned int getNumBytes(){  return m_numbytes ; }
        unsigned int getStride(){ return m_stride ; }
        unsigned long getOffset(){ return m_offset ; }
        unsigned int getCount(){  return m_count ; }
        unsigned int getSize(){   return m_size ; }  //typically 1,2,3,4 

    public:
        glm::mat4&   getModelToWorld();
        float*       getModelToWorldPtr();
        float        getExtent();
        const char*  getName();
        char         getType();

    private:
        void findBounds();
    private:
        char*        m_name   ; 
        NPY*         m_npy   ;
        void*        m_bytes   ;
        unsigned int m_size   ;   
        char         m_type ; 
        unsigned int m_numbytes ;  
        unsigned int m_stride ;  
        unsigned long m_offset ;  
        unsigned int m_count ;  

    private:
        glm::vec3*  m_low ;
        glm::vec3*  m_high ;
        glm::vec3*  m_dimensions ;
        glm::vec3*  m_center ;
        glm::mat4   m_model_to_world ; 
        float       m_extent ; 

};


inline const char* VecNPY::getName()
{
    return m_name ; 
}

inline char VecNPY::getType()
{
    return m_type ; 
}


