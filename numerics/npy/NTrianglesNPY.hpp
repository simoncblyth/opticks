#pragma once

#include <glm/glm.hpp>

template <typename T> class NPY ;
struct ntriangle ; 

class NTrianglesNPY {
    public:
        static NTrianglesNPY* hemi_octahedron();
        static NTrianglesNPY* octahedron();
        static const glm::vec3 PX ; 
        static const glm::vec3 PY ; 
        static const glm::vec3 PZ ; 
        static const glm::vec3 MX ; 
        static const glm::vec3 MY ; 
        static const glm::vec3 MZ ; 
    public:
        static NTrianglesNPY* icosahedron();
        static const glm::vec3 Ip0 ; 
        static const glm::vec3 Ip1 ; 
        static const glm::vec3 Ip2 ; 
        static const glm::vec3 Ip3 ; 
        static const glm::vec3 Ip4 ; 
        static const glm::vec3 Ip5 ;

        static const glm::vec3 Im0 ; 
        static const glm::vec3 Im1 ; 
        static const glm::vec3 Im2 ; 
        static const glm::vec3 Im3 ; 
        static const glm::vec3 Im4 ; 
        static const glm::vec3 Im5 ;
    public:
        static NTrianglesNPY* cube();
        static const glm::vec3 PXPYPZ ; 
        static const glm::vec3 PXPYMZ ; 
        static const glm::vec3 PXMYPZ ; 
        static const glm::vec3 PXMYMZ ; 
        static const glm::vec3 MXPYPZ ; 
        static const glm::vec3 MXPYMZ ; 
        static const glm::vec3 MXMYPZ ; 
        static const glm::vec3 MXMYMZ ; 
    public:
        static NTrianglesNPY* sphere(unsigned int n_polar=24, unsigned int n_azimuthal=24); 
    public:
        NTrianglesNPY(unsigned int n=0);
    public:
        NPY<float>* subdivide(unsigned int nsubdiv);
    public:
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d);
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
        void add(const ntriangle& t );
    public:
        NPY<float>* getBuffer();
        unsigned int getNumTriangles();
    private:
        NPY<float>*  m_tris ; 

};


inline NPY<float>* NTrianglesNPY::getBuffer()
{
    return m_tris ; 
}




