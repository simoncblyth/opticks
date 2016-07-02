#pragma once


#include <glm/fwd.hpp>

template <typename T> class NPY ;
class NTrianglesNPY ; 

//
//  *NTesselate*  just does triangle subdivision  
//         
//  Delaunay Tessellation is the general approach to this 
//  for a comparison of available code:
//
//  * http://library.msri.org/books/Book52/files/23liu.pdf
//
//
struct ntriangle ; 
#include "NPY_API_EXPORT.hh"

class NPY_API NTesselate {
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



