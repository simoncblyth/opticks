#pragma once

#include <glm/glm.hpp>
#include "NPY.hpp"
/*

ViewNPY is ultra-lightweight, just managing: 

* pointer to NPY
* parameters for addressing the data (stride, offset, count)
* characteristics of the data 
  (high, low, center, dimensions, extent, model2world matrix)

Many methods assume a 3 dimensional NPY array structure, 
eg with shapes like (10000,6,4) in which case j 0:5 k 0:3 size=1:4 
Trailing dimension usually 4 as quads are convenient and efficient on GPU.

*/

class ViewNPY {
    public:
        typedef enum { 
                   BYTE,   UNSIGNED_BYTE,
                   SHORT,  UNSIGNED_SHORT,
                   INT,    UNSIGNED_INT,
                   HALF_FLOAT,
                   FLOAT,
                   DOUBLE,
                   FIXED,
                   INT_2_10_10_10_REV,
                   UNSIGNED_INT_2_10_10_10_REV,
                   UNSIGNED_INT_10F_11F_11F_REV } Type_t ;
         
    public:
        ViewNPY(const char* name, NPYBase* npy, unsigned int j, unsigned int k, unsigned int l, unsigned int size=4, Type_t type=FLOAT, bool norm=false, bool iatt=false) ;
        void addressNPY();
        void setCustomOffset(unsigned long offset);
        unsigned int getValueOffset(); // ?? multiply by sizeof(att-type) to get byte offset

    public:
        void dump(const char* msg);
        void Summary(const char* msg);
        void Print(const char* msg);

        NPYBase*     getNPY(){    return m_npy   ; }
        void*        getBytes(){  return m_bytes ; }
        unsigned int getNumBytes(){  return m_numbytes ; }
        unsigned int getStride(){ return m_stride ; }
        unsigned long getOffset(){ return m_offset ; }
        unsigned int getCount(){  return m_count ; }
        unsigned int getSize(){   return m_size ; }  //typically 1,2,3,4 
        bool         getNorm(){ return m_norm ; }
        bool         getIatt(){ return m_iatt ; }
        Type_t       getType(){ return m_type ; }
        const char*  getName(){ return m_name ; }

    public:
        glm::vec4&   getCenterExtent();
        glm::mat4&   getModelToWorld();
        float*       getModelToWorldPtr();
        float        getExtent();

    private:
        void findBounds();
    private:
        char*        m_name   ; 
        NPYBase*     m_npy ; 
        void*        m_bytes   ;
    private:
        unsigned char m_j ; 
        unsigned char m_k ; 
        unsigned char m_l ; 
        unsigned int  m_size   ;   
        Type_t        m_type ; 
        bool          m_norm ;
        bool          m_iatt ;
    private:
        unsigned int  m_numbytes ;  
        unsigned int  m_stride ;  
        unsigned long m_offset ;  
        unsigned int  m_count ;  

    private:
        glm::vec3*  m_low ;
        glm::vec3*  m_high ;
        glm::vec3*  m_dimensions ;
        glm::vec3*  m_center ;
        glm::mat4   m_model_to_world ; 
        float       m_extent ; 
        glm::vec4   m_center_extent ; 
        bool        m_addressed ; 

};


inline glm::vec4& ViewNPY::getCenterExtent()
{
    return m_center_extent ; 
} 
