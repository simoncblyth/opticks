#pragma once

class NPY ; 
#include <glm/glm.hpp>

// Considered calling this SliceNPY but this 
// is much less ambitious than NumPy slicing 
// so stick with VecNPY to express the intended simplicity

class VecNPY {
    public:
        VecNPY(NPY* npy, unsigned int j, unsigned int k) ;

    public:
        void dump(const char* msg);
        void Summary(const char* msg);

        void*        getBytes(){  return m_bytes ; }
        unsigned int getNumBytes(){  return m_numbytes ; }
        unsigned int getStride(){ return m_stride ; }
        unsigned int getOffset(){ return m_offset ; }
        unsigned int getCount(){  return m_count ; }

    private:
        void findBounds();
    private:
        void*        m_bytes   ;
        //unsigned int m_size   ;    // typically 1,2,3,4 
        unsigned int m_numbytes ;  
        unsigned int m_stride ;  
        unsigned int m_offset ;  
        unsigned int m_count ;  

    private:
        glm::vec3*  m_low ;
        glm::vec3*  m_high ;
        glm::vec3*  m_dimensions ;
        glm::vec3*  m_center ;
        glm::mat4   m_model_to_world ; 
        float       m_extent ; 

};



