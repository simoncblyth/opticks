#pragma once

#include <glm/fwd.hpp>
template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"
class GGEO_API GTransforms {
    public:
        static GTransforms* make(unsigned int n);
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



