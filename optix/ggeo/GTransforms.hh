#pragma once

#include <glm/glm.hpp>
template <typename T> class NPY ;

class GTransforms {
    public:
        void save(const char* path);
        static GTransforms* load(const char* path);
        NPY<float>* getBuffer();
    public:
        GTransforms(NPY<float>* buf=NULL);
    public:
        void add(const glm::mat4& mat);
        void add();    // identity
    public:
        glm::mat4 get(unsigned int i);
    private:
        NPY<float>* m_buffer ; 

};


inline GTransforms::GTransforms(NPY<float>* buf) 
    :
    m_buffer(buf)
{
}

inline NPY<float>* GTransforms::getBuffer()
{
    return m_buffer ; 
}

