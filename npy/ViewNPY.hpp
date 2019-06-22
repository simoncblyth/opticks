#pragma once

#include <string>
#include "NGLM.hpp"

/**
ViewNPY
=========

Holds most of the paramters needed by the OpenGL data descriptor 
method from Rdr::address::

    glVertexAttribPointer(index, size, type, norm, stride, offset);


ViewNPY is ultra-lightweight, just managing: 

* pointer to NPY
* parameters for addressing the data (stride, offset, count)
* characteristics of the data 
  (high, low, center, dimensions, extent, model2world matrix)

Many methods assume a 3 dimensional NPY array structure, 
eg with shapes like (10000,6,4) in which case j 0:5 k 0:3 size=1:4 
Trailing dimension usually 4 as quads are convenient and efficient on GPU.

**/

class NPYBase ; 
class MultiViewNPY ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API ViewNPY {
    public:
        static const char* BYTE_ ; 
        static const char* UNSIGNED_BYTE_ ; 
        static const char* SHORT_ ; 
        static const char* UNSIGNED_SHORT_ ; 
        static const char* INT_ ; 
        static const char* UNSIGNED_INT_ ; 
        static const char* HALF_FLOAT_ ; 
        static const char* FLOAT_ ; 
        static const char* DOUBLE_ ; 
        static const char* FIXED_ ; 
        static const char* INT_2_10_10_10_REV_ ; 
        static const char* UNSIGNED_INT_2_10_10_10_REV_ ; 
        static const char* UNSIGNED_INT_10F_11F_11F_REV_ ;

        typedef enum { 
                   BYTE,   
                   UNSIGNED_BYTE,
                   SHORT,  
                   UNSIGNED_SHORT,
                   INT,    
                   UNSIGNED_INT,
                   HALF_FLOAT,
                   FLOAT,
                   DOUBLE,
                   FIXED,
                   INT_2_10_10_10_REV,
                   UNSIGNED_INT_2_10_10_10_REV,
                   UNSIGNED_INT_10F_11F_11F_REV } Type_t ;
         
    public:
        ViewNPY(const char* name, NPYBase* npy, unsigned int j, unsigned int k, unsigned int l, 
               unsigned int size=4, 
               Type_t type=FLOAT, 
               bool norm=false, 
               bool iatt=false, 
               unsigned int item_from_dim=1) ;

        void addressNPY();
        std::string getTypeString();
        void setCustomOffset(unsigned long offset);
        unsigned int getValueOffset(); // ?? multiply by sizeof(att-type) to get byte offset
    private:
        void init();
    public:
        void dump(const char* msg);
        void Summary(const char* msg);
        void Print(const char* msg);
        std::string description();
        std::string getShapeString();
        unsigned int getNumQuads();

        NPYBase*     getNPY();
        void*        getBytes();
        unsigned int getNumBytes();
        unsigned int getStride();
        unsigned long getOffset();
        unsigned int getCount();
        unsigned int getSize();  //typically 1,2,3,4 
        bool         getNorm();
        bool         getIatt();
        Type_t       getType();
        const char*  getTypeName();
        const char*  getName();

    public:
        glm::vec4&   getCenterExtent();
        glm::mat4 &  getModelToWorld();
        float*       getModelToWorldPtr();
        float        getExtent();

    public:
        // for debugging
        void setParent(MultiViewNPY* parent);
        MultiViewNPY* getParent();

    private:
        void findBounds();
    private:
        char*         m_name   ; 
        NPYBase*      m_npy ; 
        MultiViewNPY* m_parent ;  
        void*         m_bytes   ;
    private:
        unsigned char m_j ; 
        unsigned char m_k ; 
        unsigned char m_l ; 
        unsigned int  m_size   ;   
        Type_t        m_type ; 
        bool          m_norm ;
        bool          m_iatt ;
        unsigned int  m_item_from_dim ;   // 0-based dimension from which the item starts, preceding dimensions correspond to the count 
    private:
        unsigned int  m_numbytes ;  
        unsigned int  m_stride ;  
        unsigned long m_offset ;  

    private:
        glm::vec3*  m_low ;
        glm::vec3*  m_high ;
        glm::vec3*  m_dimensions ;
        glm::vec3*  m_center ;
        glm::mat4   m_model_to_world ; 
        glm::vec4   m_center_extent ; 
        float       m_extent ; 
        bool        m_addressed ; 

};


#include "NPY_TAIL.hh"




