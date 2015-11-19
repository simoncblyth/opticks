#pragma once

#include <cstddef>
#include <glm/glm.hpp>

template <typename T> class NPY ;
class NTrianglesNPY ; 

struct ntriangle ; 

class NTesselate {
    public:
        NTesselate(NPY<float>* basis);
        void subdivide(unsigned int nsubdiv);
        void add(glm::vec3& a, glm::vec3& c, const glm::vec3& v);
        NPY<float>* getBuffer();
    private:
        void init(); 
        void subdivide(unsigned int nsubdiv, ntriangle& t);
    private:
        NPY<float>*    m_basis ; 
        NTrianglesNPY* m_tris ; 
};


inline NTesselate::NTesselate(NPY<float>* basis) 
    :
    m_basis(basis),
    m_tris(NULL)
{
    init();
}


