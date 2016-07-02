#pragma once

#include <glm/fwd.hpp>
template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"
class GGEO_API GIds {
    public:
        static GIds* make(unsigned int n);
    public:
        static GIds* load(const char* path);
        void save(const char* path);
        NPY<unsigned int>* getBuffer();
    public:
        GIds(NPY<unsigned int>* buf=NULL);
    public:
        void add(const glm::uvec4& v);
        void add(unsigned int x, unsigned int y, unsigned int z, unsigned int w); 
    public:
        glm::uvec4 get(unsigned int i);

    private:
        NPY<unsigned int>*  m_buffer; 

};


